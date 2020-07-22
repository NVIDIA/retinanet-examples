import os
import random
from contextlib import redirect_stdout
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils import data
from pycocotools.coco import COCO
import math
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation


class CocoDataset(data.dataset.Dataset):
    'Dataset looping through a set of images'

    def __init__(self, path, resize, max_size, stride, annotations=None, training=False, rotate_augment=False,
                 augment_brightness=0.0, augment_contrast=0.0,
                 augment_hue=0.0, augment_saturation=0.0):
        super().__init__()

        self.path = os.path.expanduser(path)
        self.resize = resize
        self.max_size = max_size
        self.stride = stride
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.training = training
        self.rotate_augment = rotate_augment
        self.augment_brightness = augment_brightness
        self.augment_contrast = augment_contrast
        self.augment_hue = augment_hue
        self.augment_saturation = augment_saturation

        with redirect_stdout(None):
            self.coco = COCO(annotations)
        self.ids = list(self.coco.imgs.keys())
        if 'categories' in self.coco.dataset:
            self.categories_inv = {k: i for i, k in enumerate(self.coco.getCatIds())}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ' Get sample'

        # Load image
        id = self.ids[index]
        if self.coco:
            image = self.coco.loadImgs(id)[0]['file_name']
        im = Image.open('{}/{}'.format(self.path, image)).convert("RGB")

        # Randomly sample scale for resize during training
        resize = self.resize
        if isinstance(resize, list):
            resize = random.randint(self.resize[0], self.resize[-1])

        ratio = resize / min(im.size)
        if ratio * max(im.size) > self.max_size:
            ratio = self.max_size / max(im.size)
        im = im.resize((int(ratio * d) for d in im.size), Image.BILINEAR)

        if self.training:
            # Get annotations
            boxes, categories = self._get_target(id)
            boxes *= ratio

            # Random rotation, if self.rotate_augment
            random_angle = random.randint(0, 3) * 90
            if self.rotate_augment and random_angle != 0:
                # rotate by random_angle degrees.
                im = im.rotate(random_angle)
                x, y, w, h = boxes[:, 0].clone(), boxes[:, 1].clone(), boxes[:, 2].clone(), boxes[:, 3].clone()
                if random_angle == 90:
                    boxes[:, 0] = y - im.size[1] / 2 + im.size[0] / 2
                    boxes[:, 1] = im.size[0] / 2 + im.size[1] / 2 - x - w
                    boxes[:, 2] = h
                    boxes[:, 3] = w
                elif random_angle == 180:
                    boxes[:, 0] = im.size[0] - x - w
                    boxes[:, 1] = im.size[1] - y - h
                elif random_angle == 270:
                    boxes[:, 0] = im.size[0] / 2 + im.size[1] / 2 - y - h
                    boxes[:, 1] = x - im.size[0] / 2 + im.size[1] / 2
                    boxes[:, 2] = h
                    boxes[:, 3] = w

            # Random horizontal flip
            if random.randint(0, 1):
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, 0] = im.size[0] - boxes[:, 0] - boxes[:, 2]

            # Apply image brightness, contrast etc augmentation
            if self.augment_brightness:
                brightness_factor = random.normalvariate(1, self.augment_brightness)
                brightness_factor = max(0, brightness_factor)
                im = adjust_brightness(im, brightness_factor)
            if self.augment_contrast:
                contrast_factor = random.normalvariate(1, self.augment_contrast)
                contrast_factor = max(0, contrast_factor)
                im = adjust_contrast(im, contrast_factor)
            if self.augment_hue:
                hue_factor = random.normalvariate(0, self.augment_hue)
                hue_factor = max(-0.5, hue_factor)
                hue_factor = min(0.5, hue_factor)
                im = adjust_hue(im, hue_factor)
            if self.augment_saturation:
                saturation_factor = random.normalvariate(1, self.augment_saturation)
                saturation_factor = max(0, saturation_factor)
                im = adjust_saturation(im, saturation_factor)

            target = torch.cat([boxes, categories], dim=1)

        # Convert to tensor and normalize
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
        data = data.float().div(255).view(*im.size[::-1], len(im.mode))
        data = data.permute(2, 0, 1)

        for t, mean, std in zip(data, self.mean, self.std):
            t.sub_(mean).div_(std)

        # Apply padding
        pw, ph = ((self.stride - d % self.stride) % self.stride for d in im.size)
        data = F.pad(data, (0, pw, 0, ph))

        if self.training:
            return data, target

        return data, id, ratio

    def _get_target(self, id):
        'Get annotations for sample'

        ann_ids = self.coco.getAnnIds(imgIds=id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes, categories = [], []
        for ann in annotations:
            if ann['bbox'][2] < 1 and ann['bbox'][3] < 1:
                continue
            boxes.append(ann['bbox'])
            cat = ann['category_id']
            if 'categories' in self.coco.dataset:
                cat = self.categories_inv[cat]
            categories.append(cat)

        if boxes:
            target = (torch.FloatTensor(boxes),
                      torch.FloatTensor(categories).unsqueeze(1))
        else:
            target = (torch.ones([1, 4]), torch.ones([1, 1]) * -1)

        return target

    def collate_fn(self, batch):
        'Create batch from multiple samples'

        if self.training:
            data, targets = zip(*batch)
            max_det = max([t.size()[0] for t in targets])
            targets = [torch.cat([t, torch.ones([max_det - t.size()[0], 5]) * -1]) for t in targets]
            targets = torch.stack(targets, 0)
        else:
            data, indices, ratios = zip(*batch)

        # Pad data to match max batch dimensions
        sizes = [d.size()[-2:] for d in data]
        w, h = (max(dim) for dim in zip(*sizes))

        data_stack = []
        for datum in data:
            pw, ph = w - datum.size()[-2], h - datum.size()[-1]
            data_stack.append(
                F.pad(datum, (0, ph, 0, pw)) if max(ph, pw) > 0 else datum)

        data = torch.stack(data_stack)

        if self.training:
            return data, targets

        ratios = torch.FloatTensor(ratios).view(-1, 1, 1)
        return data, torch.IntTensor(indices), ratios


class DataIterator():
    'Data loader for data parallel'

    def __init__(self, path, resize, max_size, batch_size, stride, world, annotations, training=False,
                 rotate_augment=False, augment_brightness=0.0,
                 augment_contrast=0.0, augment_hue=0.0, augment_saturation=0.0):
        self.resize = resize
        self.max_size = max_size

        self.dataset = CocoDataset(path, resize=resize, max_size=max_size,
                                   stride=stride, annotations=annotations, training=training,
                                   rotate_augment=rotate_augment,
                                   augment_brightness=augment_brightness,
                                   augment_contrast=augment_contrast, augment_hue=augment_hue,
                                   augment_saturation=augment_saturation)
        self.ids = self.dataset.ids
        self.coco = self.dataset.coco

        self.sampler = data.distributed.DistributedSampler(self.dataset) if world > 1 else None
        self.dataloader = data.DataLoader(self.dataset, batch_size=batch_size // world,
                                          sampler=self.sampler, collate_fn=self.dataset.collate_fn, num_workers=2,
                                          pin_memory=True)

    def __repr__(self):
        return '\n'.join([
            '    loader: pytorch',
            '    resize: {}, max: {}'.format(self.resize, self.max_size),
        ])

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for output in self.dataloader:
            if self.dataset.training:
                data, target = output
            else:
                data, ids, ratio = output

            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)

            if self.dataset.training:
                if torch.cuda.is_available():
                    target = target.cuda(non_blocking=True)
                yield data, target
            else:
                if torch.cuda.is_available():
                    ids = ids.cuda(non_blocking=True)
                    ratio = ratio.cuda(non_blocking=True)
                yield data, ids, ratio


class RotatedCocoDataset(data.dataset.Dataset):
    'Dataset looping through a set of images'

    def __init__(self, path, resize, max_size, stride, annotations=None, training=False, rotate_augment=False,
                 augment_brightness=0.0, augment_contrast=0.0,
                 augment_hue=0.0, augment_saturation=0.0, absolute_angle=False):
        super().__init__()

        self.path = os.path.expanduser(path)
        self.resize = resize
        self.max_size = max_size
        self.stride = stride
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.training = training
        self.rotate_augment = rotate_augment
        self.augment_brightness = augment_brightness
        self.augment_contrast = augment_contrast
        self.augment_hue = augment_hue
        self.augment_saturation = augment_saturation
        self.absolute_angle=absolute_angle

        with redirect_stdout(None):
            self.coco = COCO(annotations)
        self.ids = list(self.coco.imgs.keys())
        if 'categories' in self.coco.dataset:
            self.categories_inv = {k: i for i, k in enumerate(self.coco.getCatIds())}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ' Get sample'

        # Load image
        id = self.ids[index]
        if self.coco:
            image = self.coco.loadImgs(id)[0]['file_name']
        im = Image.open('{}/{}'.format(self.path, image)).convert("RGB")

        # Randomly sample scale for resize during training
        resize = self.resize
        if isinstance(resize, list):
            resize = random.randint(self.resize[0], self.resize[-1])

        ratio = resize / min(im.size)
        if ratio * max(im.size) > self.max_size:
            ratio = self.max_size / max(im.size)
        im = im.resize((int(ratio * d) for d in im.size), Image.BILINEAR)

        if self.training:
            # Get annotations
            boxes, categories = self._get_target(id)
            # boxes *= ratio
            boxes[:, :4] *= ratio

            # Random rotation, if self.rotate_augment
            random_angle = random.randint(0, 3) * 90
            if self.rotate_augment and random_angle != 0:
                # rotate by random_angle degrees.
                original_size = im.size
                im = im.rotate(random_angle, expand=True)
                x, y, w, h, t = boxes[:, 0].clone(), boxes[:, 1].clone(), boxes[:, 2].clone(), \
                                boxes[:, 3].clone(), boxes[:, 4].clone()
                if random_angle == 90:
                    boxes[:, 0] = y
                    boxes[:, 1] = original_size[0] - x - w
                    if not self.absolute_angle:
                        boxes[:, 2] = h
                        boxes[:, 3] = w
                elif random_angle == 180:
                    boxes[:, 0] = original_size[0] - x - w
                    boxes[:, 1] = original_size[1] - y - h

                elif random_angle == 270:
                    boxes[:, 0] = original_size[1] - y - h
                    boxes[:, 1] = x
                    if not self.absolute_angle:
                        boxes[:, 2] = h
                        boxes[:, 3] = w

                    pass

                # Adjust theta
                if self.absolute_angle:
                    # This is only needed in absolute angle mode.
                    t += math.radians(random_angle)
                    rem = torch.remainder(torch.abs(t), math.pi)
                    sign = torch.sign(t)
                    t = rem * sign

                boxes[:, 4] = t

            # Random horizontal flip
            if random.randint(0, 1):
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, 0] = im.size[0] - boxes[:, 0] - boxes[:, 2]
                boxes[:, 1] = boxes[:, 1]
                boxes[:, 4] = -boxes[:, 4]

            # Apply image brightness, contrast etc augmentation
            if self.augment_brightness:
                brightness_factor = random.normalvariate(1, self.augment_brightness)
                brightness_factor = max(0, brightness_factor)
                im = adjust_brightness(im, brightness_factor)
            if self.augment_contrast:
                contrast_factor = random.normalvariate(1, self.augment_contrast)
                contrast_factor = max(0, contrast_factor)
                im = adjust_contrast(im, contrast_factor)
            if self.augment_hue:
                hue_factor = random.normalvariate(0, self.augment_hue)
                hue_factor = max(-0.5, hue_factor)
                hue_factor = min(0.5, hue_factor)
                im = adjust_hue(im, hue_factor)
            if self.augment_saturation:
                saturation_factor = random.normalvariate(1, self.augment_saturation)
                saturation_factor = max(0, saturation_factor)
                im = adjust_saturation(im, saturation_factor)

            target = torch.cat([boxes, categories], dim=1)

        # Convert to tensor and normalize
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
        data = data.float().div(255).view(*im.size[::-1], len(im.mode))
        data = data.permute(2, 0, 1)

        for t, mean, std in zip(data, self.mean, self.std):
            t.sub_(mean).div_(std)

        # Apply padding
        pw, ph = ((self.stride - d % self.stride) % self.stride for d in im.size)
        data = F.pad(data, (0, pw, 0, ph))

        if self.training:
            return data, target

        return data, id, ratio

    def _get_target(self, id):
        'Get annotations for sample'

        ann_ids = self.coco.getAnnIds(imgIds=id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes, categories = [], []
        for ann in annotations:
            if ann['bbox'][2] < 1 and ann['bbox'][3] < 1:
                continue
            final_bbox = ann['bbox']
            if len(final_bbox) == 4:
                final_bbox.append(0.0)  # add theta of zero.
            assert len(ann['bbox']) == 5, "Bounding box for id %i does not contain five entries." % id
            boxes.append(final_bbox)
            cat = ann['category_id']
            if 'categories' in self.coco.dataset:
                cat = self.categories_inv[cat]
            categories.append(cat)

        if boxes:
            target = (torch.FloatTensor(boxes),
                      torch.FloatTensor(categories).unsqueeze(1))
        else:
            target = (torch.ones([1, 5]), torch.ones([1, 1]) * -1)

        return target

    def collate_fn(self, batch):
        'Create batch from multiple samples'

        if self.training:
            data, targets = zip(*batch)
            max_det = max([t.size()[0] for t in targets])
            targets = [torch.cat([t, torch.ones([max_det - t.size()[0], 6]) * -1]) for t in targets]
            targets = torch.stack(targets, 0)
        else:
            data, indices, ratios = zip(*batch)

        # Pad data to match max batch dimensions
        sizes = [d.size()[-2:] for d in data]
        w, h = (max(dim) for dim in zip(*sizes))

        data_stack = []
        for datum in data:
            pw, ph = w - datum.size()[-2], h - datum.size()[-1]
            data_stack.append(
                F.pad(datum, (0, ph, 0, pw)) if max(ph, pw) > 0 else datum)

        data = torch.stack(data_stack)

        if self.training:
            return data, targets

        ratios = torch.FloatTensor(ratios).view(-1, 1, 1)
        return data, torch.IntTensor(indices), ratios


class RotatedDataIterator():
    'Data loader for data parallel'

    def __init__(self, path, resize, max_size, batch_size, stride, world, annotations, training=False,
                 rotate_augment=False, augment_brightness=0.0,
                 augment_contrast=0.0, augment_hue=0.0, augment_saturation=0.0, absolute_angle=False
                 ):
        self.resize = resize
        self.max_size = max_size

        self.dataset = RotatedCocoDataset(path, resize=resize, max_size=max_size,
                                          stride=stride, annotations=annotations, training=training,
                                          rotate_augment=rotate_augment,
                                          augment_brightness=augment_brightness,
                                          augment_contrast=augment_contrast, augment_hue=augment_hue,
                                          augment_saturation=augment_saturation, absolute_angle=absolute_angle)
        self.ids = self.dataset.ids
        self.coco = self.dataset.coco

        self.sampler = data.distributed.DistributedSampler(self.dataset) if world > 1 else None
        self.dataloader = data.DataLoader(self.dataset, batch_size=batch_size // world,
                                          sampler=self.sampler, collate_fn=self.dataset.collate_fn, num_workers=2,
                                          pin_memory=True)

    def __repr__(self):
        return '\n'.join([
            '    loader: pytorch',
            '    resize: {}, max: {}'.format(self.resize, self.max_size),
        ])

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for output in self.dataloader:
            if self.dataset.training:
                data, target = output
            else:
                data, ids, ratio = output

            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)

            if self.dataset.training:
                if torch.cuda.is_available():
                    target = target.cuda(non_blocking=True)
                yield data, target
            else:
                if torch.cuda.is_available():
                    ids = ids.cuda(non_blocking=True)
                    ratio = ratio.cuda(non_blocking=True)
                yield data, ids, ratio
