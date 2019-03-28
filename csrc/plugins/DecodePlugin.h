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

#include <cassert>
#include <vector>

#include "../cuda/decode.h"

using namespace nvinfer1;

#define RETINANET_PLUGIN_NAME "RetinaNetDecode"
#define RETINANET_PLUGIN_VERSION "1"
#define RETINANET_PLUGIN_NAMESPACE ""

namespace retinanet {

class DecodePlugin : public IPluginV2 {
  float _score_thresh;
  int _top_n;
  std::vector<float> _anchors;
  float _scale;

  size_t _height;
  size_t _width;
  size_t _num_anchors;
  size_t _num_classes;

protected:
  void deserialize(void const* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    read(d, _score_thresh);
    read(d, _top_n);
    size_t anchors_size;
    read(d, anchors_size);
    while( anchors_size-- ) {
      float val;
      read(d, val);
      _anchors.push_back(val);
    }
    read(d, _scale);
    read(d, _height);
    read(d, _width);
    read(d, _num_anchors);
    read(d, _num_classes);
  }

  size_t getSerializationSize() const override {
    return sizeof(_score_thresh) + sizeof(_top_n)
      + sizeof(size_t) + sizeof(float) * _anchors.size() + sizeof(_scale)
      + sizeof(_height) + sizeof(_width) + sizeof(_num_anchors) + sizeof(_num_classes);
  }

  void serialize(void *buffer) const override {
    char* d = static_cast<char*>(buffer);
    write(d, _score_thresh);
    write(d, _top_n);
    write(d, _anchors.size());
    for( auto &val : _anchors ) {
      write(d, val);
    }
    write(d, _scale);
    write(d, _height);
    write(d, _width);
    write(d, _num_anchors);
    write(d, _num_classes);
  }

public:
  DecodePlugin(float score_thresh, int top_n, std::vector<float> const& anchors, int scale)
    : _score_thresh(score_thresh), _top_n(top_n), _anchors(anchors), _scale(scale) {}

  DecodePlugin(void const* data, size_t length) {
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
    assert(nbInputDims == 2);
    assert(index < this->getNbOutputs());
    return Dims3(_top_n * (index == 1 ? 4 : 1), 1, 1);
  }

  bool supportsFormat(DataType type, PluginFormat format) const override {
    return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
  }

  void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, 
                        int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {
    assert(type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW);
    assert(nbInputs == 2);
    auto const& scores_dims = inputDims[0];
    auto const& boxes_dims = inputDims[1];
    assert(scores_dims.d[1] == boxes_dims.d[1]);
    assert(scores_dims.d[2] == boxes_dims.d[2]);
    _height = scores_dims.d[1];
    _width = scores_dims.d[2];
    _num_anchors = boxes_dims.d[0] / 4; 
    _num_classes = scores_dims.d[0] / _num_anchors;
  }

  int initialize() override { return 0; }

  void terminate() override {}

  size_t getWorkspaceSize(int maxBatchSize) const override {
    static int size = -1;
    if (size < 0) {
      size = cuda::decode(maxBatchSize, nullptr, nullptr, _height, _width, _scale,
        _num_anchors, _num_classes, _anchors, _score_thresh, _top_n, 
        nullptr, 0, nullptr);
    }
    return size;
  }

  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override {
    return cuda::decode(batchSize, inputs, outputs, _height, _width, _scale,
      _num_anchors, _num_classes, _anchors, _score_thresh, _top_n,
      workspace, getWorkspaceSize(batchSize), stream);
  }

  void destroy() override {};

  const char *getPluginNamespace() const override {
    return RETINANET_PLUGIN_NAMESPACE;
  }
  
  void setPluginNamespace(const char *N) override {

  }

  IPluginV2 *clone() const override {
    return new DecodePlugin(_score_thresh, _top_n, _anchors, _scale);
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

class DecodePluginCreator : public IPluginCreator {
public:
  DecodePluginCreator() {}

  const char *getPluginName () const override {
    return RETINANET_PLUGIN_NAME;
  }

  const char *getPluginVersion () const override {
    return RETINANET_PLUGIN_VERSION;
  }
 
  const char *getPluginNamespace() const override {
    return RETINANET_PLUGIN_NAMESPACE;
  }

  IPluginV2 *deserializePlugin (const char *name, const void *serialData, size_t serialLength) override {
    return new DecodePlugin(serialData, serialLength);
  }

  void setPluginNamespace(const char *N) override {}
  const PluginFieldCollection *getFieldNames() override { return nullptr; }
  IPluginV2 *createPlugin (const char *name, const PluginFieldCollection *fc) override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);

}

#undef RETINANET_PLUGIN_NAME
#undef RETINANET_PLUGIN_VERSION
#undef RETINANET_PLUGIN_NAMESPACE
