import os
import json
import tempfile
from contextlib import redirect_stdout
import torch
from apex import amp
from apex.parallel import DistributedDataParallel as ADDP
from torch.nn.parallel import DistributedDataParallel
from pycocotools.cocoeval import COCOeval
import numpy as np

from .data import DataIterator, RotatedDataIterator
from .dali import DaliDataIterator
from .model import Model
from .utils import Profiler, rotate_box


def infer(model, path, detections_file, resize, max_size, batch_size, mixed_precision=True, is_master=True, world=0,
          annotations=None, no_apex=False, use_dali=True, is_validation=False, verbose=True, rotated_bbox=False):
    'Run inference on images from path'

    DDP = DistributedDataParallel if no_apex else ADDP
    backend = 'pytorch' if isinstance(model, Model) or isinstance(model, DDP) else 'tensorrt'

    stride = model.module.stride if isinstance(model, DDP) else model.stride

    # Create annotations if none was provided
    if not annotations:
        annotations = tempfile.mktemp('.json')
        images = [{'id': i, 'file_name': f} for i, f in enumerate(os.listdir(path))]
        json.dump({'images': images}, open(annotations, 'w'))

    # TensorRT only supports fixed input sizes, so override input size accordingly
    if backend == 'tensorrt': max_size = max(model.input_size)

    # Prepare dataset
    if verbose: print('Preparing dataset...')
    if rotated_bbox:
        if use_dali: raise NotImplementedError("This repo does not currently support DALI for rotated bbox.")
        data_iterator = RotatedDataIterator(path, resize, max_size, batch_size, stride,
                                            world, annotations, training=False)
    else:
        data_iterator = (DaliDataIterator if use_dali else DataIterator)(
            path, resize, max_size, batch_size, stride,
            world, annotations, training=False)
    if verbose: print(data_iterator)

    # Prepare model
    if backend == 'pytorch':
        # If we are doing validation during training,
        # no need to register model with AMP again
        if not is_validation:
            if torch.cuda.is_available(): model = model.to(memory_format=torch.channels_last).cuda()
            if not no_apex:
                model = amp.initialize(model, None,
                                    opt_level='O2' if mixed_precision else 'O0',
                                    keep_batchnorm_fp32=True,
                                    verbosity=0)

        model.eval()

    if verbose:
        print('   backend: {}'.format(backend))
        print('    device: {} {}'.format(
            world, 'cpu' if not torch.cuda.is_available() else 'GPU' if world == 1 else 'GPUs'))
        print('     batch: {}, precision: {}'.format(batch_size,
                                                     'unknown' if backend == 'tensorrt' else 'mixed' if mixed_precision else 'full'))
        print(' BBOX type:', 'rotated' if rotated_bbox else 'axis aligned')
        print('Running inference...')

    results = []
    profiler = Profiler(['infer', 'fw'])
    with torch.no_grad():
        for i, (data, ids, ratios) in enumerate(data_iterator):
            # Forward pass
            if backend=='pytorch': data = data.contiguous(memory_format=torch.channels_last)
            profiler.start('fw')
            scores, boxes, classes = model(data, rotated_bbox) #Need to add model size (B, 3, W, H)
            profiler.stop('fw')

            results.append([scores, boxes, classes, ids, ratios])

            profiler.bump('infer')
            if verbose and (profiler.totals['infer'] > 60 or i == len(data_iterator) - 1):
                size = len(data_iterator.ids)
                msg = '[{:{len}}/{}]'.format(min((i + 1) * batch_size,
                                                 size), size, len=len(str(size)))
                msg += ' {:.3f}s/{}-batch'.format(profiler.means['infer'], batch_size)
                msg += ' (fw: {:.3f}s)'.format(profiler.means['fw'])
                msg += ', {:.1f} im/s'.format(batch_size / profiler.means['infer'])
                print(msg, flush=True)

                profiler.reset()

    # Gather results from all devices
    if verbose: print('Gathering results...')
    results = [torch.cat(r, dim=0) for r in zip(*results)]
    if world > 1:
        for r, result in enumerate(results):
            all_result = [torch.ones_like(result, device=result.device) for _ in range(world)]
            torch.distributed.all_gather(list(all_result), result)
            results[r] = torch.cat(all_result, dim=0)

    if is_master:
        # Copy buffers back to host
        results = [r.cpu() for r in results]

        # Collect detections
        detections = []
        processed_ids = set()
        for scores, boxes, classes, image_id, ratios in zip(*results):
            image_id = image_id.item()
            if image_id in processed_ids:
                continue
            processed_ids.add(image_id)

            keep = (scores > 0).nonzero(as_tuple=False)
            scores = scores[keep].view(-1)
            if rotated_bbox:
                boxes = boxes[keep, :].view(-1, 6)
                boxes[:, :4] /= ratios
            else:
                boxes = boxes[keep, :].view(-1, 4) / ratios
            classes = classes[keep].view(-1).int()

            for score, box, cat in zip(scores, boxes, classes):
                if rotated_bbox:
                    x1, y1, x2, y2, sin, cos = box.data.tolist()
                    theta = np.arctan2(sin, cos)
                    w = x2 - x1 + 1
                    h = y2 - y1 + 1
                    seg = rotate_box([x1, y1, w, h, theta])
                else:
                    x1, y1, x2, y2 = box.data.tolist()
                cat = cat.item()
                if 'annotations' in data_iterator.coco.dataset:
                    cat = data_iterator.coco.getCatIds()[cat]
                this_det = {
                    'image_id': image_id,
                    'score': score.item(),
                    'category_id': cat}
                if rotated_bbox:
                    this_det['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1, theta]
                    this_det['segmentation'] = [seg]
                else:
                    this_det['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

                detections.append(this_det)

        if detections:
            # Save detections
            if detections_file and verbose: print('Writing {}...'.format(detections_file))
            detections = {'annotations': detections}
            detections['images'] = data_iterator.coco.dataset['images']
            if 'categories' in data_iterator.coco.dataset:
                detections['categories'] = data_iterator.coco.dataset['categories']
            if detections_file:
                for d_file in detections_file:
                    json.dump(detections, open(d_file, 'w'), indent=4)

            # Evaluate model on dataset
            if 'annotations' in data_iterator.coco.dataset:
                if verbose: print('Evaluating model...')
                with redirect_stdout(None):
                    coco_pred = data_iterator.coco.loadRes(detections['annotations'])
                    if rotated_bbox:
                        coco_eval = COCOeval(data_iterator.coco, coco_pred, 'segm')
                    else:
                        coco_eval = COCOeval(data_iterator.coco, coco_pred, 'bbox')
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                coco_eval.summarize()
                return coco_eval.stats # mAP and mAR
        else:
            print('No detections!')
            return None
    return 0
