import os.path
import time
import json
import warnings
import signal
from datetime import datetime
from contextlib import contextmanager
from PIL import Image, ImageDraw
import requests

def show_detections(detections):
    'Show image with drawn detections'

    for image, detections in detections.items():
        im = Image.open(image).convert('RGBA')
        overlay = Image.new('RGBA', im.size, (255,255,255,0))
        draw = ImageDraw.Draw(overlay)
        detections.sort(key=lambda d: d['score'])
        for detection in detections:
            box = detection['bbox']
            alpha = int(detection['score'] * 255)
            draw.rectangle(box, outline=(255, 255, 255, alpha))
            draw.text((box[0]+2, box[1]), '[{}]'.format(detection['class']),
                fill=(255, 255, 255, alpha))
            draw.text((box[0]+2, box[1]+10), '{:.2}'.format(detection['score']),
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
        self.lasts = { k: 0 for k in names }
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
                data={ 'time': int(datetime.now().timestamp() * 1e9), 
                        'metric': k, 'value': v })
    except Exception as e:
        warnings.warn('Warning: posting metrics failed: {}'.format(e))

