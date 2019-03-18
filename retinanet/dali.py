import os
from contextlib import redirect_stdout
from math import ceil
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import random
from nvidia.dali import pipeline, ops, types
from pycocotools.coco import COCO

class COCOPipeline(pipeline.Pipeline):
    'Dali pipeline for COCO'

    def __init__(self, batch_size, num_threads, ids, path, coco, training):
        super().__init__(batch_size, num_threads, torch.cuda.current_device(), prefetch_queue_depth=num_threads)

        self.path = path
        self.training = training
        self.ids = ids
        self.coco = coco
        self.iter = 0

        # Create operators
        self.input = ops.ExternalSource()
        self.input_ids = ops.ExternalSource()
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

    def define_graph(self):
        # Feed image through decode
        self.images = self.input()
        self.images_ids = self.input_ids()
        images = self.decode(self.images)
        return images, self.images_ids

    def iter_setup(self):
        # Get next COCO images for the batch
        images, ids = [], []
        overflow = False
        for _ in range(self.batch_size):
            id = int(self.ids[self.iter])
            file_name = self.coco.loadImgs(id)[0]['file_name']
            image = open(self.path + file_name, 'rb')
            images.append(np.frombuffer(image.read(), dtype=np.uint8))
            ids.append(np.array([-1 if overflow else id], dtype=np.float))
            
            overflow = self.iter + 1 >= len(self.ids)
            if not overflow:
                self.iter = (self.iter + 1) % len(self.ids)
    
        self.feed_input(self.images, images)
        self.feed_input(self.images_ids, ids)

class DaliDataIterator():
    'Data loader for data parallel using Dali'

    def __init__(self, path, resize, max_size, batch_size, stride, world, annotations, training=False):
        self.training = training
        self.resize = resize
        self.max_size = max_size
        self.stride = stride
        self.batch_size = batch_size // world
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Setup COCO
        with redirect_stdout(None):
            self.coco = COCO(annotations)
        self.ids = list(self.coco.imgs.keys())
        if 'categories' in self.coco.dataset:
            self.categories_inv = { k: i for i, k in enumerate(self.coco.getCatIds()) }
        self.local_ids = np.array_split(np.array(self.ids), world)[torch.cuda.current_device()]

        self.pipe = COCOPipeline(batch_size=self.batch_size, num_threads=2, 
            ids=self.local_ids, path=path, coco=self.coco, training=training)
        self.pipe.build()

    def __repr__(self):
        return '\n'.join([
            '    loader: dali',
            '    resize: {}, max: {}'.format(self.resize, self.max_size),
        ])

    def __len__(self):
        return ceil(len(self.local_ids) / self.batch_size)

    def __iter__(self):
        for _ in range(self.__len__()):
            data, ratios, ids = [], [], []
            dali_data, dali_ids = self.pipe.run()
            for batch in range(self.batch_size):
                id = int(dali_ids.at(batch)[0])
                if id < 0: break
                
                # Convert dali tensor to pytorch
                dali_tensor = dali_data.at(batch)
                datum = torch.zeros(dali_tensor.shape(), dtype=torch.uint8, device=torch.device('cuda'))
                c_type_pointer = ctypes.c_void_p(datum.data_ptr())
                dali_tensor.copy_to_external(c_type_pointer)

                # Normalize tensor
                datum = datum.float().div(255).permute(2, 0, 1).unsqueeze(0)
                for t, mean, std in zip(datum, self.mean, self.std):
                    t.sub_(mean).div_(std)

                # Randomly sample scale for resize during training
                resize = self.resize
                if isinstance(resize, list):
                    resize = random.randint(self.resize[0], self.resize[-1])

                size = datum.shape[-2:]
                ratio = resize / min(size)
                if ratio * max(size) > self.max_size:
                    ratio = self.max_size / max(size)
                datum = F.interpolate(datum, scale_factor=ratio, mode='bilinear', align_corners=False)

                ids.append(id)
                data.append(datum)
                ratios.append(ratio)

            # Pad batch data
            sizes = [max([data[b].shape[2+i] for b, _ in enumerate(data)]) for i in range(2)]
            sizes = [ceil(s / self.stride) * self.stride for s in sizes]
            for i, datum in enumerate(data):
                data[i] = F.pad(datum, (0, sizes[1] - datum.size(3), 0, sizes[0] - datum.size(2)))
            data = torch.cat(data, dim=0)

            ids = torch.Tensor(ids).int().cuda()
            ratios = torch.Tensor(ratios).cuda()

            yield data, ids, ratios
