FROM nvcr.io/nvidian/pytorch:19.07-py3

COPY . retinanet/
RUN pip install --no-cache-dir -e retinanet/
