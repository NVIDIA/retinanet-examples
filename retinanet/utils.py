import os.path
import time
import json
import warnings
import signal
from datetime import datetime
from contextlib import contextmanager
from PIL import Image, ImageDraw
import requests
import numpy as np
import math
import torch

def order_points(pts):
    pts_reorder = []

    for idx, pt in enumerate(pts):
        idx = torch.argsort(pt[:, 0])
        xSorted = pt[idx, :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[torch.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        D = torch.cdist(tl[np.newaxis], rightMost)[0]
        (br, tr) = rightMost[torch.argsort(D, descending=True), :]
        pts_reorder.append(torch.stack([tl, tr, br, bl]))

    return torch.stack([p for p in pts_reorder])

def rotate_boxes(boxes, points=False):
    
    #Rotate target bounding boxes 
    
    #Input:  
    #    Target boxes (xmin_ymin, width_height, theta)
    #Output:
    #    boxes_axis (xmin_ymin, xmax_ymax, sin(theta), cos(theta))
    #    boxes_rotated (xy0, xy1, xy2, xy3)
    
    boxes, theta = boxes.split(4, dim=1)
    theta = theta.squeeze(1)
    
    if points:
        boxes = torch.stack([boxes[:,0],boxes[:,1], 
            boxes[:,2], boxes[:,1], 
            boxes[:,2], boxes[:,3], 
            boxes[:,0], boxes[:,3]],1).view(-1,4,2)
    else:
        boxes = torch.stack([boxes[:,0],boxes[:,1], 
                (boxes[:,0]+boxes[:,2]), boxes[:,1], 
                (boxes[:,0]+boxes[:,2]), (boxes[:,1]+boxes[:,3]), 
                boxes[:,0], (boxes[:,1]+boxes[:,3])],1).view(-1,4,2)
    
    boxes_axis = torch.cat([torch.stack([boxes[:,0,:], boxes[:,2,:]],1).view(-1,4), 
            torch.sin(theta[:, None]), torch.cos(theta[:, None])],1)

    u = torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1)
    l = torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)
    R = torch.stack([u, l], dim=1)
    cents = torch.mean(boxes, 1)[:,None,:].repeat(1,4,1)
    #boxes_rotated = order_points(torch.matmul(boxes - cents, R) + cents).view(-1,8)
    boxes_rotated = order_points(torch.matmul(R,(boxes - cents).transpose(1,2)).transpose(1,2) + cents).view(-1,8)
    
    return boxes_axis, boxes_rotated



def rotate_box(bbox):
    xmin, ymin, width, height, theta = bbox

    xy1 = xmin, ymin
    xy2 = xmin, ymin + height - 1
    xy3 = xmin + width - 1, ymin + height - 1
    xy4 = xmin + width - 1, ymin

    cents = np.array([xmin + (width - 1) / 2, ymin + (height - 1) / 2])

    corners = np.stack([xy1, xy2, xy3, xy4])

    u = np.stack([np.cos(theta), -np.sin(theta)])
    l = np.stack([np.sin(theta), np.cos(theta)])
    R = np.vstack([u, l])

    corners = np.matmul(R, (corners - cents).transpose(1, 0)).transpose(1, 0) + cents

    return corners.reshape(-1).tolist()


def show_detections(detections):
    'Show image with drawn detections'

    for image, detections in detections.items():
        im = Image.open(image).convert('RGBA')
        overlay = Image.new('RGBA', im.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        detections.sort(key=lambda d: d['score'])
        for detection in detections:
            box = detection['bbox']
            alpha = int(detection['score'] * 255)
            draw.rectangle(box, outline=(255, 255, 255, alpha))
            draw.text((box[0] + 2, box[1]), '[{}]'.format(detection['class']),
                      fill=(255, 255, 255, alpha))
            draw.text((box[0] + 2, box[1] + 10), '{:.2}'.format(detection['score']),
                      fill=(255, 255, 255, alpha))
        im = Image.alpha_composite(im, overlay)
        im.show()


def save_detections(path, detections):
    print('Writing detections to {}...'.format(os.path.basename(path)))
    with open(path, 'w') as f:
        json.dump(detections, f)


@contextmanager
def ignore_sigint():
    handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, handler)


class Profiler(object):
    def __init__(self, names=['main']):
        self.names = names
        self.lasts = {k: 0 for k in names}
        self.totals = self.lasts.copy()
        self.counts = self.lasts.copy()
        self.means = self.lasts.copy()
        self.reset()

    def reset(self):
        last = time.time()
        for name in self.names:
            self.lasts[name] = last
            self.totals[name] = 0
            self.counts[name] = 0
            self.means[name] = 0

    def start(self, name='main'):
        self.lasts[name] = time.time()

    def stop(self, name='main'):
        self.totals[name] += time.time() - self.lasts[name]
        self.counts[name] += 1
        self.means[name] = self.totals[name] / self.counts[name]

    def bump(self, name='main'):
        self.stop(name)
        self.start(name)


def post_metrics(url, metrics):
    try:
        for k, v in metrics.items():
            requests.post(url,
                          data={'time': int(datetime.now().timestamp() * 1e9),
                                'metric': k, 'value': v})
    except Exception as e:
        warnings.warn('Warning: posting metrics failed: {}'.format(e))
