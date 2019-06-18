import os
import json
import tempfile
from contextlib import redirect_stdout
import torch
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .data import DataIterator
from .dali import DaliDataIterator
from .model import Model
from .utils import Profiler

def infer(model, path, detections_file, resize, max_size, batch_size, mixed_precision=True, is_master=True, world=0, annotations=None, use_dali=True, is_validation=False, verbose=True):
    'Run inference on images from path'

    backend = 'pytorch' if isinstance(model, Model) or isinstance(model, DDP) else 'tensorrt'

    stride = model.module.stride if isinstance(model, DDP) else model.stride

    # Create annotations if none was provided
    if not annotations:
        annotations = tempfile.mktemp('.json')
        images = [{ 'id': i, 'file_name': f} for i, f in enumerate(os.listdir(path))]
        json.dump({ 'images': images }, open(annotations, 'w'))

    # TensorRT only supports fixed input sizes, so override input size accordingly
    if backend == 'tensorrt': max_size = max(model.input_size)

    # Prepare dataset
    if verbose: print('Preparing dataset...')
    data_iterator = (DaliDataIterator if use_dali else DataIterator)(
        path, resize, max_size, batch_size, stride,
        world, annotations, training=False)
    if verbose: print(data_iterator)

    # Prepare model
    if backend is 'pytorch':
        # If we are doing validation during training,
        # no need to register model with AMP again
        if not is_validation:
            if torch.cuda.is_available(): model = model.cuda()
            model = amp.initialize(model, None,
                               opt_level = 'O2' if mixed_precision else 'O0',
                               keep_batchnorm_fp32 = True,
                               verbosity = 0)

        model.eval()

    if verbose:
        print('   backend: {}'.format(backend))
        print('    device: {} {}'.format(
            world, 'cpu' if not torch.cuda.is_available() else 'gpu' if world == 1 else 'gpus'))
        print('     batch: {}, precision: {}'.format(batch_size,
            'unknown' if backend is 'tensorrt' else 'mixed' if mixed_precision else 'full'))
        print('Running inference...')

    results = []
    profiler = Profiler(['infer', 'fw'])
    with torch.no_grad():
        for i, (data, ids, ratios) in enumerate(data_iterator):
            # Forward pass
            profiler.start('fw')
            scores, boxes, classes = model(data)
            profiler.stop('fw')

            results.append([scores, boxes, classes, ids, ratios])

            profiler.bump('infer')
            if verbose and (profiler.totals['infer'] > 60 or i == len(data_iterator) - 1):
                size = len(data_iterator.ids)
                msg  = '[{:{len}}/{}]'.format(min((i + 1) * batch_size,
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

            keep = (scores > 0).nonzero()
            scores = scores[keep].view(-1)
            boxes = boxes[keep, :].view(-1, 4) / ratios
            classes = classes[keep].view(-1).int()

            for score, box, cat in zip(scores, boxes, classes):
                x1, y1, x2, y2 = box.data.tolist()
                cat = cat.item()
                if 'annotations' in data_iterator.coco.dataset:
                    cat = data_iterator.coco.getCatIds()[cat]
                detections.append({
                    'image_id': image_id,
                    'score': score.item(),
                    'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                    'category_id': cat
                })

        if detections:
            # Save detections
            if detections_file and verbose: print('Writing {}...'.format(detections_file))
            detections = { 'annotations': detections }
            detections['images'] = data_iterator.coco.dataset['images']
            if 'categories' in data_iterator.coco.dataset:
                detections['categories'] = [data_iterator.coco.dataset['categories']]
            if detections_file:
                json.dump(detections, open(detections_file, 'w'), indent=4)

            # Evaluate model on dataset
            if 'annotations' in data_iterator.coco.dataset:
                if verbose: print('Evaluating model...')
                with redirect_stdout(None):
                    coco_pred = data_iterator.coco.loadRes(detections['annotations'])
                    coco_eval = COCOeval(data_iterator.coco, coco_pred, 'bbox')
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                coco_eval.summarize()
        else:
            print('No detections!')
