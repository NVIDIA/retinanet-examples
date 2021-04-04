#!/usr/bin/env python3
import sys
import os
import argparse
import random
import torch.cuda
import torch.distributed
import torch.multiprocessing

from odtk import infer, train, utils
from odtk.model import Model
from odtk._C import Engine


def parse(args):
    parser = argparse.ArgumentParser(description='ODTK: Object Detection Toolkit.')
    parser.add_argument('--master', metavar='address:port', type=str, help='Address and port of the master worker',
                        default='127.0.0.1:29500')

    subparsers = parser.add_subparsers(help='sub-command', dest='command')
    subparsers.required = True

    devcount = max(1, torch.cuda.device_count())

    parser_train = subparsers.add_parser('train', help='train a network')
    parser_train.add_argument('model', type=str, help='path to output model or checkpoint to resume from')
    parser_train.add_argument('--annotations', metavar='path', type=str, help='path to COCO style annotations',
                              required=True)
    parser_train.add_argument('--images', metavar='path', type=str, help='path to images', default='.')
    parser_train.add_argument('--backbone', action='store', type=str, nargs='+', help='backbone model (or list of)',
                              default=['ResNet50FPN'])
    parser_train.add_argument('--classes', metavar='num', type=int, help='number of classes', default=80)
    parser_train.add_argument('--batch', metavar='size', type=int, help='batch size', default=2 * devcount)
    parser_train.add_argument('--resize', metavar='scale', type=int, help='resize to given size', default=800)
    parser_train.add_argument('--max-size', metavar='max', type=int, help='maximum resizing size', default=1333)
    parser_train.add_argument('--jitter', metavar='min max', type=int, nargs=2, help='jitter size within range',
                              default=[640, 1024])
    parser_train.add_argument('--iters', metavar='number', type=int, help='number of iterations to train for',
                              default=90000)
    parser_train.add_argument('--milestones', action='store', type=int, nargs='*',
                              help='list of iteration indices where learning rate decays', default=[60000, 80000])
    parser_train.add_argument('--schedule', metavar='scale', type=float,
                              help='scale schedule (affecting iters and milestones)', default=1)
    parser_train.add_argument('--full-precision', help='train in full precision', action='store_true')
    parser_train.add_argument('--lr', metavar='value', help='learning rate', type=float, default=0.01)
    parser_train.add_argument('--warmup', metavar='iterations', help='numer of warmup iterations', type=int,
                              default=1000)
    parser_train.add_argument('--gamma', metavar='value', type=float,
                              help='multiplicative factor of learning rate decay', default=0.1)
    parser_train.add_argument('--override', help='override model', action='store_true')
    parser_train.add_argument('--val-annotations', metavar='path', type=str,
                              help='path to COCO style validation annotations')
    parser_train.add_argument('--val-images', metavar='path', type=str, help='path to validation images')
    parser_train.add_argument('--post-metrics', metavar='url', type=str, help='post metrics to specified url')
    parser_train.add_argument('--fine-tune', metavar='path', type=str, help='fine tune a pretrained model')
    parser_train.add_argument('--logdir', metavar='logdir', type=str, help='directory where to write logs')
    parser_train.add_argument('--val-iters', metavar='number', type=int,
                              help='number of iterations between each validation', default=8000)
    parser_train.add_argument('--no-apex', help='use Pytorch native AMP and DDP', action='store_true')
    parser_train.add_argument('--with-dali', help='use dali for data loading', action='store_true')
    parser_train.add_argument('--augment-rotate', help='use four-fold rotational augmentation', action='store_true')
    parser_train.add_argument('--augment-free-rotate', type=float, metavar='value value', nargs=2, default=[0, 0],
                              help='rotate images by an arbitrary angle, between min and max (in degrees)')
    parser_train.add_argument('--augment-brightness', metavar='value', type=float,
                              help='adjust the brightness of the image.', default=0.002)
    parser_train.add_argument('--augment-contrast', metavar='value', type=float,
                              help='adjust the contrast of the image.', default=0.002)
    parser_train.add_argument('--augment-hue', metavar='value', type=float,
                              help='adjust the hue of the image.', default=0.0002)
    parser_train.add_argument('--augment-saturation', metavar='value', type=float,
                              help='adjust the saturation of the image.', default=0.002)
    parser_train.add_argument('--regularization-l2', metavar='value', type=float, help='L2 regularization for optim',
                              default=0.0001)
    parser_train.add_argument('--rotated-bbox', help='detect rotated bounding boxes [x, y, w, h, theta]',
                              action='store_true')
    parser_train.add_argument('--anchor-ious', metavar='value value', type=float, nargs=2,
                              help='anchor/bbox overlap threshold', default=[0.4, 0.5])
    parser_train.add_argument('--absolute-angle', help='regress absolute angle (rather than -45 to 45 degrees.',
                              action='store_true')

    parser_infer = subparsers.add_parser('infer', help='run inference')
    parser_infer.add_argument('model', type=str, help='path to model')
    parser_infer.add_argument('--images', metavar='path', type=str, help='path to images', default='.')
    parser_infer.add_argument('--annotations', metavar='annotations', type=str,
                              help='evaluate using provided annotations')
    parser_infer.add_argument('--output', metavar='file', type=str, nargs='+',
                              help='save detections to specified JSON file(s)', default=['detections.json'])
    parser_infer.add_argument('--batch', metavar='size', type=int, help='batch size', default=2 * devcount)
    parser_infer.add_argument('--resize', metavar='scale', type=int, help='resize to given size', default=800)
    parser_infer.add_argument('--max-size', metavar='max', type=int, help='maximum resizing size', default=1333)
    parser_infer.add_argument('--no-apex', help='use Pytorch native AMP and DDP', action='store_true')
    parser_infer.add_argument('--with-dali', help='use dali for data loading', action='store_true')
    parser_infer.add_argument('--full-precision', help='inference in full precision', action='store_true')
    parser_infer.add_argument('--rotated-bbox', help='inference using a rotated bounding box model',
                              action='store_true')

    parser_export = subparsers.add_parser('export', help='export a model into a TensorRT engine')
    parser_export.add_argument('model', type=str, help='path to model')
    parser_export.add_argument('export', type=str, help='path to exported output')
    parser_export.add_argument('--size', metavar='height width', type=int, nargs='+',
                               help='input size (square) or sizes (h w) to use when generating TensorRT engine',
                               default=[1280])
    parser_export.add_argument('--full-precision', help='export in full instead of half precision', action='store_true')
    parser_export.add_argument('--int8', help='calibrate model and export in int8 precision', action='store_true')
    parser_export.add_argument('--calibration-batches', metavar='size', type=int,
                               help='number of batches to use for int8 calibration', default=2)
    parser_export.add_argument('--calibration-images', metavar='path', type=str,
                               help='path to calibration images to use for int8 calibration', default="")
    parser_export.add_argument('--calibration-table', metavar='path', type=str,
                               help='path of existing calibration table to load from, or name of new calibration table',
                               default="")
    parser_export.add_argument('--verbose', help='enable verbose logging', action='store_true')
    parser_export.add_argument('--rotated-bbox', help='inference using a rotated bounding box model',
                               action='store_true')
    parser_export.add_argument('--dynamic-batch-opts', help='Profile batch sizes for tensorrt engine export (min, opt, max)',
                               metavar='value value value', type=int, nargs=3, default=[1,8,16])

    return parser.parse_args(args)


def load_model(args, verbose=False):
    if args.command != 'train' and not os.path.isfile(args.model):
        raise RuntimeError('Model file {} does not exist!'.format(args.model))

    model = None
    state = {}
    _, ext = os.path.splitext(args.model)

    if args.command == 'train' and (not os.path.exists(args.model) or args.override):
        if verbose: print('Initializing model...')
        model = Model(backbones=args.backbone, classes=args.classes, rotated_bbox=args.rotated_bbox,
                      anchor_ious=args.anchor_ious)
        model.initialize(args.fine_tune)
        # Freeze unused params from training
        for n, p in model.named_parameters():
            if any(i in n for i in model.unused_modules):
                p.requires_grad = False
        if verbose: print(model)

    elif ext == '.pth' or ext == '.torch':
        if verbose: print('Loading model from {}...'.format(os.path.basename(args.model)))
        model, state = Model.load(filename=args.model, rotated_bbox=args.rotated_bbox)
        if verbose: print(model)

    elif args.command == 'infer' and ext in ['.engine', '.plan']:
        model = None

    else:
        raise RuntimeError('Invalid model format "{}"!'.format(ext))

    state['path'] = args.model
    return model, state


def worker(rank, args, world, model, state):
    'Per-device distributed worker'

    if torch.cuda.is_available():
        os.environ.update({
            'MASTER_PORT': args.master.split(':')[-1],
            'MASTER_ADDR': ':'.join(args.master.split(':')[:-1]),
            'WORLD_SIZE': str(world),
            'RANK': str(rank),
            'CUDA_DEVICE': str(rank)
        })

        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        if (args.command != 'export') and (args.batch % world != 0):
            raise RuntimeError('Batch size should be a multiple of the number of GPUs')

    if model and model.angles is not None:
        args.rotated_bbox = True

    if args.command == 'train':
        train.train(model, state, args.images, args.annotations,
                    args.val_images or args.images, args.val_annotations, args.resize, args.max_size, args.jitter,
                    args.batch, int(args.iters * args.schedule), args.val_iters, not args.full_precision, args.lr,
                    args.warmup, [int(m * args.schedule) for m in args.milestones], args.gamma,
                    rank, world=world, no_apex=args.no_apex, use_dali=args.with_dali,
                    metrics_url=args.post_metrics, logdir=args.logdir, verbose=(rank == 0),
                    rotate_augment=args.augment_rotate,
                    augment_brightness=args.augment_brightness, augment_contrast=args.augment_contrast,
                    augment_hue=args.augment_hue, augment_saturation=args.augment_saturation,
                    regularization_l2=args.regularization_l2, rotated_bbox=args.rotated_bbox, absolute_angle=args.absolute_angle)

    elif args.command == 'infer':
        if model is None:
            if rank == 0: print('Loading CUDA engine from {}...'.format(os.path.basename(args.model)))
            model = Engine.load(args.model)

        infer.infer(model, args.images, args.output, args.resize, args.max_size, args.batch,
                    annotations=args.annotations, mixed_precision=not args.full_precision,
                    is_master=(rank == 0), world=world, no_apex=args.no_apex, use_dali=args.with_dali,
                    verbose=(rank == 0), rotated_bbox=args.rotated_bbox)

    elif args.command == 'export':
        onnx_only = args.export.split('.')[-1] == 'onnx'
        input_size = args.size * 2 if len(args.size) == 1 else args.size

        calibration_files = []
        if args.int8:
            # Get list of images to use for calibration
            if os.path.isdir(args.calibration_images):
                import glob
                file_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
                for ex in file_extensions:
                    calibration_files += glob.glob("{}/*{}".format(args.calibration_images, ex), recursive=True)
                # Only need enough images for specified num of calibration batches
                if len(calibration_files) >= args.calibration_batches * args.dynamic_batch_opts[1]:
                    calibration_files = calibration_files[:(args.calibration_batches * args.dynamic_batch_opts[1])]
                else:
                    # Number of images for calibration must be greater than or equal to the kOPT optimization profile
                    if len(calibration_files) >= args.dynamic_batch_opts[1]:
                        print('Only found enough images for {} batches. Continuing anyway...'.format(
                            len(calibration_files) // args.dynamic_batch_opts[1]))
                    else:
                        raise RuntimeError('Not enough images found for calibration. ({} < {})'
                                            .format(len(calibration_files), args.dynamic_batch_opts[1]))

                random.shuffle(calibration_files)

        precision = "FP32"
        if args.int8:
            precision = "INT8"
        elif not args.full_precision:
            precision = "FP16"

        exported = model.export(input_size, args.dynamic_batch_opts, precision, calibration_files,
                                args.calibration_table, args.verbose, onnx_only=onnx_only)
        if onnx_only:
            with open(args.export, 'wb') as out:
                out.write(exported)
        else:
            exported.save(args.export)


def main(args=None):
    'Entry point for the odtk command'

    args = parse(args or sys.argv[1:])

    model, state = load_model(args, verbose=True)
    if model: model.share_memory()

    world = torch.cuda.device_count()
    if args.command == 'export' or world <= 1:
        worker(0, args, 1, model, state)
    else:
        torch.multiprocessing.spawn(worker, args=(args, world, model, state), nprocs=world)


if __name__ == '__main__':
    main()
