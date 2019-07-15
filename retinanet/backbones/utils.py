import sys
import torchvision

def register_torchvision_030(f):
    if torchvision.__version__ > '0.2.1':
        return register(f)

def register(f):
    all = sys.modules[f.__module__].__dict__.setdefault('__all__', [])
    if f.__name__ in all:
        raise RuntimeError('{} already exist!'.format(f.__name__))
    all.append(f.__name__)
    return f
