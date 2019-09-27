import os.path
import io
import math
import torch
import torch.nn as nn

from . import backbones as backbones_mod
from ._C import Engine
from .box import generate_anchors, snap_to_anchors, decode, nms
from .loss import FocalLoss, SmoothL1Loss

class Model(nn.Module):
    'RetinaNet - https://arxiv.org/abs/1708.02002'

    def __init__(self, backbones='ResNet50FPN', classes=80, config={}):
        super().__init__()

        if not isinstance(backbones, list):
            backbones = [backbones]

        self.backbones = nn.ModuleDict({b: getattr(backbones_mod, b)() for b in backbones})
        self.name = 'RetinaNet'
        self.exporting = False

        self.ratios = [1.0, 2.0, 0.5]
        self.scales = [4 * 2**(i/3) for i in range(3)]
        self.anchors = {}
        self.classes = classes

        self.threshold  = config.get('threshold', 0.05)
        self.top_n      = config.get('top_n', 1000)
        self.nms        = config.get('nms', 0.5)
        self.detections = config.get('detections', 100)

        self.stride = max([b.stride for _, b in self.backbones.items()])

        # classification and box regression heads
        def make_head(out_size):
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]
            layers += [nn.Conv2d(256, out_size, 3, padding=1)]
            return nn.Sequential(*layers)

        anchors = len(self.ratios) * len(self.scales)
        self.cls_head = make_head(classes * anchors)
        self.box_head = make_head(4 * anchors)

        self.cls_criterion = FocalLoss()
        self.box_criterion = SmoothL1Loss(beta=0.11)

    def __repr__(self):
        return '\n'.join([
            '     model: {}'.format(self.name),
            '  backbone: {}'.format(', '.join([k for k, _ in self.backbones.items()])),
            '   classes: {}, anchors: {}'.format(self.classes, len(self.ratios) * len(self.scales)),
        ])

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = { k: v for k, v in chk['state_dict'].items() if k not in ignored }
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbone(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)

        # Initialize class head prior
        def initialize_prior(layer):
            pi = 0.01
            b = - math.log((1 - pi) / pi)
            nn.init.constant_(layer.bias, b)
            nn.init.normal_(layer.weight, std=0.01)
        self.cls_head[-1].apply(initialize_prior)

    def forward(self, x):
        if self.training: x, targets = x

        # Backbones forward pass
        features = []
        for _, backbone in self.backbones.items():
            features.extend(backbone(x))

        # Heads forward pass
        cls_heads = [self.cls_head(t) for t in features]
        box_heads = [self.box_head(t) for t in features]

        if self.training:
            return self._compute_loss(x, cls_heads, box_heads, targets.float())

        cls_heads = [cls_head.sigmoid() for cls_head in cls_heads]

        if self.exporting:
            self.strides = [x.shape[-1] // cls_head.shape[-1] for cls_head in cls_heads]
            return cls_heads, box_heads

        # Inference post-processing
        decoded = []
        for cls_head, box_head in zip(cls_heads, box_heads):
            # Generate level's anchors
            stride = x.shape[-1] // cls_head.shape[-1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales)

            # Decode and filter boxes
            decoded.append(decode(cls_head, box_head, stride,
                self.threshold, self.top_n, self.anchors[stride]))

        # Perform non-maximum suppression
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        return nms(*decoded, self.nms, self.detections)

    def _extract_targets(self, targets, stride, size):
        cls_target, box_target, depth = [], [], []
        for target in targets:
            target = target[target[:, -1] > -1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales)
            snapped = snap_to_anchors(
                target, [s * stride for s in size[::-1]], stride,
                self.anchors[stride].to(targets.device), self.classes, targets.device)
            for l, s in zip((cls_target, box_target, depth), snapped): l.append(s)
        return torch.stack(cls_target), torch.stack(box_target), torch.stack(depth)

    def _compute_loss(self, x, cls_heads, box_heads, targets):
        cls_losses, box_losses, fg_targets = [], [], []
        for cls_head, box_head in zip(cls_heads, box_heads):
            size = cls_head.shape[-2:]
            stride = x.shape[-1] / cls_head.shape[-1]

            cls_target, box_target, depth = self._extract_targets(targets, stride, size)
            fg_targets.append((depth > 0).sum().float().clamp(min=1))

            cls_head = cls_head.view_as(cls_target).float()
            cls_mask = (depth >= 0).expand_as(cls_target).float()
            cls_loss = self.cls_criterion(cls_head, cls_target)
            cls_loss = cls_mask * cls_loss
            cls_losses.append(cls_loss.sum())

            box_head = box_head.view_as(box_target).float()
            box_mask = (depth > 0).expand_as(box_target).float()
            box_loss = self.box_criterion(box_head, box_target)
            box_loss = box_mask * box_loss
            box_losses.append(box_loss.sum())

        fg_targets = torch.stack(fg_targets).sum()
        cls_loss = torch.stack(cls_losses).sum() / fg_targets
        box_loss = torch.stack(box_losses).sum() / fg_targets
        return cls_loss, box_loss

    def save(self, state):
        checkpoint = {
            'backbone': [k for k, _ in self.backbones.items()],
            'classes': self.classes,
            'state_dict': self.state_dict()
        }

        for key in ('iteration', 'optimizer', 'scheduler'):
            if key in state:
                checkpoint[key] = state[key]

        torch.save(checkpoint, state['path'])

    @classmethod
    def load(cls, filename):
        if not os.path.isfile(filename):
            raise ValueError('No checkpoint {}'.format(filename))

        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        # Recreate model from checkpoint instead of from individual backbones
        model = cls(backbones=checkpoint['backbone'], classes=checkpoint['classes'])
        model.load_state_dict(checkpoint['state_dict'])

        state = {}
        for key in ('iteration', 'optimizer', 'scheduler'):
            if key in checkpoint:
                state[key] = checkpoint[key]

        del checkpoint
        torch.cuda.empty_cache()

        return model, state

    def export(self, size, batch, precision, calibration_files, calibration_table, verbose, onnx_only=False):

        import torch.onnx.symbolic_opset9 as onnx_symbolic
        def upsample_nearest2d(g, input, output_size):
            # Currently, TRT 5.1/6.0 ONNX Parser does not support all ONNX ops
            # needed to support dynamic upsampling ONNX forumlation
            # Here we hardcode scale=2 as a temporary workaround
            scales = g.op("Constant", value_t=torch.tensor([1.,1.,2.,2.]))
            return g.op("Upsample", input, scales, mode_s="nearest")

        onnx_symbolic.upsample_nearest2d = upsample_nearest2d
 
        # Export to ONNX
        print('Exporting to ONNX...')
        self.exporting = True
        onnx_bytes = io.BytesIO()
        zero_input = torch.zeros([1, 3, *size]).cuda()
        extra_args = { 'verbose': verbose }
        torch.onnx.export(self.cuda(), zero_input, onnx_bytes, *extra_args)
        self.exporting = False

        if onnx_only:
            return onnx_bytes.getvalue()

        # Build TensorRT engine
        model_name = '_'.join([k for k, _ in self.backbones.items()])
        anchors = [generate_anchors(stride, self.ratios, self.scales).view(-1).tolist() 
            for stride in self.strides]
        return Engine(onnx_bytes.getvalue(), len(onnx_bytes.getvalue()), batch, precision,
            self.threshold, self.top_n, anchors, self.nms, self.detections, calibration_files, model_name, calibration_table, verbose)
