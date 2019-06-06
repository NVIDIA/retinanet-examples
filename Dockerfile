FROM nvcr.io/nvidia/pytorch:19.05-py3

COPY . retinanet/
RUN pip install --no-cache-dir -e retinanet/
