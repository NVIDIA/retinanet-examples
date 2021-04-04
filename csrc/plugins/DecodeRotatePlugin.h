/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "../cuda/decode_rotate.h"

using namespace nvinfer1;

#define RETINANET_PLUGIN_NAME "RetinaNetDecodeRotate"
#define RETINANET_PLUGIN_VERSION "1"
#define RETINANET_PLUGIN_NAMESPACE ""

namespace odtk {

class DecodeRotatePlugin : public IPluginV2DynamicExt {
  float _score_thresh;
  int _top_n;
  std::vector<float> _anchors;
  float _scale;

  size_t _height;
  size_t _width;
  size_t _num_anchors;
  size_t _num_classes;
  mutable int size = -1;

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
  DecodeRotatePlugin(float score_thresh, int top_n, std::vector<float> const& anchors, int scale)
    : _score_thresh(score_thresh), _top_n(top_n), _anchors(anchors), _scale(scale) {}

  DecodeRotatePlugin(float score_thresh, int top_n, std::vector<float> const& anchors, int scale,
    size_t height, size_t width, size_t num_anchors, size_t num_classes)
    : _score_thresh(score_thresh), _top_n(top_n), _anchors(anchors), _scale(scale),
      _height(height), _width(width), _num_anchors(num_anchors), _num_classes(num_classes) {}

  DecodeRotatePlugin(void const* data, size_t length) {
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

  DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs,
    int nbInputs, IExprBuilder &exprBuilder) override 
  {
    DimsExprs output(inputs[0]);
    output.d[1] = exprBuilder.constant(_top_n * (outputIndex == 1 ? 6 : 1));
    output.d[2] = exprBuilder.constant(1);
    output.d[3] = exprBuilder.constant(1);

    return output;
  }


  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, 
    int nbInputs, int nbOutputs) override
  {
    assert(nbInputs == 2);
    assert(nbOutputs == 3);
    assert(pos < 5);
    return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR;
  }


  int initialize() override { return 0; }

  void terminate() override {}

  size_t getWorkspaceSize(const PluginTensorDesc *inputs, 
    int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const override 
  {
    if (size < 0) {
      size = cuda::decode_rotate(inputs->dims.d[0], nullptr, nullptr, _height, _width, _scale,
        _num_anchors, _num_classes, _anchors, _score_thresh, _top_n, 
        nullptr, 0, nullptr);
    }
    return size;
  }

  int enqueue(const PluginTensorDesc *inputDesc, 
    const PluginTensorDesc *outputDesc, const void *const *inputs, 
    void *const *outputs, void *workspace, cudaStream_t stream) override 
  {
    return cuda::decode_rotate(inputDesc->dims.d[0], inputs, outputs, _height, _width, _scale,
      _num_anchors, _num_classes, _anchors, _score_thresh, _top_n,
      workspace, getWorkspaceSize(inputDesc, 2, outputDesc, 3), stream);
  }

  void destroy() override {
    delete this;
  };

  const char *getPluginNamespace() const override {
    return RETINANET_PLUGIN_NAMESPACE;
  }

  void setPluginNamespace(const char *N) override {}

  DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
  {
    assert(index < 3);
    return DataType::kFLOAT;
  }

  void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, 
    const DynamicPluginTensorDesc *out, int nbOutputs)
  {
    assert(nbInputs == 2);
    assert(nbOutputs == 3);
    auto const& scores_dims = in[0].desc.dims;
    auto const& boxes_dims = in[1].desc.dims;
    assert(scores_dims.d[2] == boxes_dims.d[2]);
    assert(scores_dims.d[3] == boxes_dims.d[3]);
    _height = scores_dims.d[2];
    _width = scores_dims.d[3];
    _num_anchors = boxes_dims.d[1] / 6; 
    _num_classes = scores_dims.d[1] / _num_anchors;
  }

  IPluginV2DynamicExt *clone() const override {
    return new DecodeRotatePlugin(_score_thresh, _top_n, _anchors, _scale, _height, _width, 
      _num_anchors, _num_classes);
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

class DecodeRotatePluginCreator : public IPluginCreator {
public:
  DecodeRotatePluginCreator() {}

  const char *getPluginName () const override {
    return RETINANET_PLUGIN_NAME;
  }

  const char *getPluginVersion () const override {
    return RETINANET_PLUGIN_VERSION;
  }
 
  const char *getPluginNamespace() const override {
    return RETINANET_PLUGIN_NAMESPACE;
  }

  IPluginV2DynamicExt *deserializePlugin (const char *name, const void *serialData, size_t serialLength) override {
    return new DecodeRotatePlugin(serialData, serialLength);
  }

  void setPluginNamespace(const char *N) override {}
  const PluginFieldCollection *getFieldNames() override { return nullptr; }
  IPluginV2DynamicExt *createPlugin (const char *name, const PluginFieldCollection *fc) override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(DecodeRotatePluginCreator);

}

#undef RETINANET_PLUGIN_NAME
#undef RETINANET_PLUGIN_VERSION
#undef RETINANET_PLUGIN_NAMESPACE
