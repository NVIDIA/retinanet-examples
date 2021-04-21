FROM nvcr.io/nvidia/pytorch:21.02-py3

COPY . odtk/
RUN pip install --no-cache-dir -e odtk/
