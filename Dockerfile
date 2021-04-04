FROM nvcr.io/nvidia/pytorch:20.11-py3

COPY . odtk/
RUN pip install --no-cache-dir -e odtk/
