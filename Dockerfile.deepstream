FROM nvcr.io/nvidia/pytorch:19.02-py3

COPY . /workspace/retinanet-examples/

RUN apt-get update && apt-get install -y libssl1.0.0  libgstreamer1.0-0    gstreamer1.0-tools   gstreamer1.0-plugins-good   gstreamer1.0-plugins-bad     gstreamer1.0-plugins-ugly  gstreamer1.0-libav   libgstrtspserver-1.0-0   libjansson4 ffmpeg

WORKDIR /root

RUN git clone https://github.com/edenhill/librdkafka.git /librdkafka && \
    cd /librdkafka && ./configure && make && make install && \
    mkdir -p /usr/local/deepstream && \
    cp /usr/local/lib/librdkafka* /usr/local/deepstream

COPY extras/deepstream/DeepStream_Release/binaries.tbz2  \
     extras/deepstream/DeepStream_Release/LicenseAgreement.pdf  \
     extras/deepstream/DeepStream_Release/README \
     /root/DeepStream_Release/

RUN cd /root/DeepStream_Release && \
    tar -xvf binaries.tbz2 -C /

# config files + sample apps
COPY extras/deepstream/DeepStream_Release/samples  \
     /root/DeepStream_Release/samples

COPY extras/deepstream/DeepStream_Release/sources \
     /root/DeepStream_Release/sources

RUN  chmod u+x /root/DeepStream_Release/sources/tools/nvds_logger/setup_nvds_logger.sh

# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video

RUN ln -sf /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so

RUN pip install --no-cache-dir -e /workspace/retinanet-examples

RUN mkdir /workspace/retinanet-examples/extras/deepstream/deepstream-sample/build && \
    cd /workspace/retinanet-examples/extras/deepstream/deepstream-sample/build && \
    cmake -DDeepStream_DIR=/root/DeepStream_Release .. && make

WORKDIR /workspace/retinanet-examples/extras/deepstream
