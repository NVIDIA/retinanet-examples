import os
from contextlib import redirect_stdout
from math import ceil
import ctypes
import numpy as np
import torch
import numpy as np
from nvidia.dali import pipeline, ops, types
from pycocotools.coco import COCO

class COCOPipeline(pipeline.Pipeline):
    'Dali pipeline for COCO'

    def __init__(self, batch_size, num_threads, path, training, annotations, world, device_id, mean, std, resize, max_size, stride):
        super().__init__(batch_size=batch_size, num_threads=num_threads, device_id = device_id, prefetch_queue_depth=num_threads, seed=42)

        self.path = path
        self.training = training
        self.stride = stride
        self.iter = 0

        self.reader = ops.COCOReader(annotations_file=annotations, file_root=path, num_shards=world,shard_id=torch.cuda.current_device(), 
                                     ltrb=True, ratio=True, shuffle_after_epoch=True, save_img_ids=True)

        self.decode_train = ops.ImageDecoderSlice(device="mixed", output_type=types.RGB)
        self.decode_infer = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.bbox_crop = ops.RandomBBoxCrop(device='cpu', ltrb=True, scaling=[0.3, 1.0], thresholds=[0.1,0.3,0.5,0.7,0.9])

        self.bbox_flip = ops.BbFlip(device='cpu', ltrb=True)
        self.img_flip = ops.Flip(device='gpu')
        self.coin_flip = ops.CoinFlip(probability=0.5)

        if isinstance(resize, list): resize = max(resize)
        self.rand_resize = ops.Uniform(range=[resize, float(max_size)])

        self.resize_train = ops.Resize(device='gpu', interp_type=types.DALIInterpType.INTERP_CUBIC, save_attrs=True)
        self.resize_infer = ops.Resize(device='gpu', interp_type=types.DALIInterpType.INTERP_CUBIC, resize_longer=max_size, save_attrs=True)

        padded_size = max_size + ((self.stride - max_size % self.stride) % self.stride)

        self.pad = ops.Paste(device='gpu', fill_value = 0, ratio=1.1, min_canvas_size=padded_size, paste_x=0, paste_y=0)
        self.normalize = ops.CropMirrorNormalize(device='gpu', mean=mean, std=std, crop=(padded_size, padded_size), crop_pos_x=0, crop_pos_y=0)

    def define_graph(self):

        images, bboxes, labels, img_ids = self.reader()

        if self.training:
            crop_begin, crop_size, bboxes, labels = self.bbox_crop(bboxes, labels)
            images = self.decode_train(images, crop_begin, crop_size)
            resize = self.rand_resize()
            images, attrs = self.resize_train(images, resize_longer=resize)

            flip = self.coin_flip()
            bboxes = self.bbox_flip(bboxes, horizontal=flip)
            images = self.img_flip(images, horizontal=flip)

        else:
            images = self.decode_infer(images)
            images, attrs = self.resize_infer(images)

        resized_images = images
        images = self.normalize(self.pad(images))

        return images, bboxes, labels, img_ids, attrs, resized_images


class DaliDataIterator():
    'Data loader for data parallel using Dali'

    def __init__(self, path, resize, max_size, batch_size, stride, world, annotations, training=False):
        self.training = training
        self.resize = resize
        self.max_size = max_size
        self.stride = stride
        self.batch_size = batch_size // world
        self.mean = [255.*x for x in [0.485, 0.456, 0.406]]
        self.std = [255.*x for x in [0.229, 0.224, 0.225]]
        self.world = world
        self.path = path

        # Setup COCO
        with redirect_stdout(None):
            self.coco = COCO(annotations)
        self.ids = list(self.coco.imgs.keys())
        if 'categories' in self.coco.dataset:
            self.categories_inv = { k: i for i, k in enumerate(self.coco.getCatIds()) }

        self.pipe = COCOPipeline(batch_size=self.batch_size, num_threads=2, 
            path=path, training=training, annotations=annotations, world=world, 
            device_id = torch.cuda.current_device(), mean=self.mean, std=self.std, resize=resize, max_size=max_size, stride=self.stride)

        self.pipe.build()

    def __repr__(self):
        return '\n'.join([
            '    loader: dali',
            '    resize: {}, max: {}'.format(self.resize, self.max_size),
        ])

    def __len__(self):
        return ceil(len(self.ids) // self.world / self.batch_size)

    def __iter__(self):
        for _ in range(self.__len__()):

            data, ratios, ids, num_detections = [], [], [], []
            dali_data, dali_boxes, dali_labels, dali_ids, dali_attrs, dali_resize_img = self.pipe.run()

            for l in range(len(dali_boxes)):
                num_detections.append(dali_boxes.at(l).shape[0])

            pyt_targets = -1 * torch.ones([len(dali_boxes), max(max(num_detections),1), 5])

            for batch in range(self.batch_size):
                id = int(dali_ids.at(batch)[0])
                
                # Convert dali tensor to pytorch
                dali_tensor = dali_data.at(batch)
                tensor_shape = dali_tensor.shape()

                datum = torch.zeros(dali_tensor.shape(), dtype=torch.float, device=torch.device('cuda'))
                c_type_pointer = ctypes.c_void_p(datum.data_ptr())
                dali_tensor.copy_to_external(c_type_pointer)

                # Calculate image resize ratio to rescale boxes
                prior_size = dali_attrs.as_cpu().at(batch)
                resized_size = dali_resize_img.at(batch).shape()
                ratio = max(resized_size) / max(prior_size)

                if self.training:
                    # Rescale boxes
                    b_arr = dali_boxes.at(batch)
                    num_dets = b_arr.shape[0]
                    if num_dets is not 0:
                        pyt_bbox = torch.from_numpy(b_arr).float()

                        pyt_bbox[:,0] *= float(prior_size[1])
                        pyt_bbox[:,1] *= float(prior_size[0])
                        pyt_bbox[:,2] *= float(prior_size[1])
                        pyt_bbox[:,3] *= float(prior_size[0])
                        # (l,t,r,b) ->  (x,y,w,h) == (l,r, r-l, b-t)
                        pyt_bbox[:,2] -= pyt_bbox[:,0]
                        pyt_bbox[:,3] -= pyt_bbox[:,1]
                        pyt_targets[batch,:num_dets,:4] = pyt_bbox * ratio

                    # Arrange labels in target tensor
                    l_arr = dali_labels.at(batch)
                    if num_dets is not 0:
                        pyt_label = torch.from_numpy(l_arr).float()
                        pyt_label -= 1 #Rescale labels to [0,79] instead of [1,80]
                        pyt_targets[batch,:num_dets, 4] = pyt_label.squeeze()

                ids.append(id)
                data.append(datum.unsqueeze(0))
                ratios.append(ratio)

            data = torch.cat(data, dim=0)

            if self.training:
                pyt_targets = pyt_targets.cuda(non_blocking=True)

                yield data, pyt_targets

            else:
                ids = torch.Tensor(ids).int().cuda(non_blocking=True)
                ratios = torch.Tensor(ratios).cuda(non_blocking=True)

                yield data, ids, ratios
