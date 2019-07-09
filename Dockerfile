FROM nvcr.io/nvidian/pytorch:19.07-py3

RUN pip uninstall --y torchvision

RUN git clone https://github.com/pytorch/vision.git && cd vision && git checkout c94a158 && python setup.py install

COPY . retinanet/
RUN pip install --no-cache-dir -e retinanet/
