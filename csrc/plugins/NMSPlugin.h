/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <NvInfer.h>

#include <vector>
#include <cassert>

#include "../cuda/nms.h"

using namespace nvinfer1;

#define RETINANET_PLUGIN_NAME "RetinaNetNMS"
#define RETINANET_PLUGIN_VERSION "1"
#define RETINANET_PLUGIN_NAMESPACE ""

namespace retinanet {

class NMSPlugin : public IPluginV2 {
  float _nms_thresh;
  int _detections_per_im;

  size_t _count;

protected:
  void deserialize(void const* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    read(d, _nms_thresh);
    read(d, _detections_per_im);
    read(d, _count);
  }

  size_t getSerializationSize() const override {
    return sizeof(_nms_thresh) + sizeof(_detections_per_im)
      + sizeof(_count);
  }

  void serialize(void *buffer) const override {
    char* d = static_cast<char*>(buffer);
    write(d, _nms_thresh);
    write(d, _detections_per_im);
    write(d, _count);
  }

public:
  NMSPlugin(float nms_thresh, int detections_per_im)
    : _nms_thresh(nms_thresh), _detections_per_im(detections_per_im) {
    assert(nms_thresh > 0);
    assert(detections_per_im > 0);
  }

  NMSPlugin(void const* data, size_t length) {
    this->deserialize(data, length);
  }

  const char *getPluginType() const override {
    return RETINANET_PLUGIN_NAME;
  }
 
  const char *getPluginVersion() const override {
    return RETINANET_PLUGIN_VERSION;
  }
  
  int getNbOutputs() const override {
    return 3;
  }

  Dims getOutputDimensions(int index,
                                     const Dims *inputs, int nbInputDims) override {
    assert(nbInputDims == 3);
    assert(index < this->getNbOutputs());
    return Dims3(_detections_per_im * (index == 1 ? 4 : 1), 1, 1);
  }

  bool supportsFormat(DataType type, PluginFormat format) const override {
    return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
  }

  void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, 
                        int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {
    assert(type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW);
    assert(nbInputs == 3);
    assert(inputDims[0].d[0] == inputDims[2].d[0]);
    assert(inputDims[1].d[0] == inputDims[2].d[0] * 4);
    _count = inputDims[0].d[0];
  }

  int initialize() override { return 0; }

  void terminate() override {}

  size_t getWorkspaceSize(int maxBatchSize) const override {
    static int size = -1;
    if (size < 0) {
      size = cuda::nms(maxBatchSize, nullptr, nullptr, _count, 
        _detections_per_im, _nms_thresh, 
        nullptr, 0, nullptr);
    }
    return size;
  }

  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override {
    return cuda::nms(batchSize, inputs, outputs, _count, 
      _detections_per_im, _nms_thresh,
      workspace, getWorkspaceSize(batchSize), stream);
  }

  void destroy() override {}

  const char *getPluginNamespace() const override {
    return RETINANET_PLUGIN_NAMESPACE;
  }
  
  void setPluginNamespace(const char *N) override {
    
  }

  IPluginV2 *clone() const override {
    return new NMSPlugin(_nms_thresh, _detections_per_im);
  }

private:
  template<typename T> void write(char*& buffer, const T& val) const {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
  }

  template<typename T> void read(const char*& buffer, T& val) {
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
  }
};

class NMSPluginCreator : public IPluginCreator {
public:
  NMSPluginCreator() {}
  
  const char *getPluginNamespace() const override {
    return RETINANET_PLUGIN_NAMESPACE;
  }
  const char *getPluginName () const override {
    return RETINANET_PLUGIN_NAME;
  }

  const char *getPluginVersion () const override {
    return RETINANET_PLUGIN_VERSION;
  }
 
  IPluginV2 *deserializePlugin (const char *name, const void *serialData, size_t serialLength) override {
    return new NMSPlugin(serialData, serialLength);
  }

  void setPluginNamespace(const char *N) override {}
  const PluginFieldCollection *getFieldNames() override { return nullptr; }
  IPluginV2 *createPlugin (const char *name, const PluginFieldCollection *fc) override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(NMSPluginCreator);

}

#undef RETINANET_PLUGIN_NAME
#undef RETINANET_PLUGIN_VERSION
#undef RETINANET_PLUGIN_NAMESPACE
