FROM nvcr.io/nvidia/pytorch:20.02-py3

COPY . retinanet/
RUN pip install --no-cache-dir -e retinanet/
