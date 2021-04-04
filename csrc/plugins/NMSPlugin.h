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

#include <vector>
#include <cassert>

#include "../cuda/nms.h"

using namespace nvinfer1;

#define RETINANET_PLUGIN_NAME "RetinaNetNMS"
#define RETINANET_PLUGIN_VERSION "1"
#define RETINANET_PLUGIN_NAMESPACE ""

namespace odtk {

class NMSPlugin : public IPluginV2DynamicExt {
  float _nms_thresh;
  int _detections_per_im;

  size_t _count;
  mutable int size = -1;

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

  NMSPlugin(float nms_thresh, int detections_per_im, size_t count)
    : _nms_thresh(nms_thresh), _detections_per_im(detections_per_im), _count(count) {
    assert(nms_thresh > 0);
    assert(detections_per_im > 0);
    assert(count > 0);
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

  DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs,
    int nbInputs, IExprBuilder &exprBuilder) override 
  {
    DimsExprs output(inputs[0]);
    output.d[1] = exprBuilder.constant(_detections_per_im * (outputIndex == 1 ? 4 : 1));
    output.d[2] = exprBuilder.constant(1);
    output.d[3] = exprBuilder.constant(1);
    return output;
  }

  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, 
    int nbInputs, int nbOutputs) override
  {
    assert(nbInputs == 3);
    assert(nbOutputs == 3);
    assert(pos < 6);
    return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR;
  }

  int initialize() override { return 0; }

  void terminate() override {}

  size_t getWorkspaceSize(const PluginTensorDesc *inputs, 
    int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const override 
  {
    if (size < 0) {
      size = cuda::nms(inputs->dims.d[0], nullptr, nullptr, _count, 
        _detections_per_im, _nms_thresh, 
        nullptr, 0, nullptr);
    }
    return size;
  }

  int enqueue(const PluginTensorDesc *inputDesc, 
    const PluginTensorDesc *outputDesc, const void *const *inputs, 
    void *const *outputs, void *workspace, cudaStream_t stream) 
  {
    return cuda::nms(inputDesc->dims.d[0], inputs, outputs, _count, 
      _detections_per_im, _nms_thresh,
      workspace, getWorkspaceSize(inputDesc, 3, outputDesc, 3), stream);
  }

  void destroy() override {
    delete this;
  }

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
    assert(nbInputs == 3);
    assert(in[0].desc.dims.d[1] == in[2].desc.dims.d[1]);
    assert(in[1].desc.dims.d[1] == in[2].desc.dims.d[1] * 4);
    _count = in[0].desc.dims.d[1];
  }

  IPluginV2DynamicExt *clone() const {
    return new NMSPlugin(_nms_thresh, _detections_per_im, _count);
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
 
  //Was IPluginV2
  IPluginV2DynamicExt *deserializePlugin (const char *name, const void *serialData, size_t serialLength) override {
    return new NMSPlugin(serialData, serialLength);
  }
  
  //Was IPluginV2
  void setPluginNamespace(const char *N) override {}
  const PluginFieldCollection *getFieldNames() override { return nullptr; }
  IPluginV2DynamicExt *createPlugin (const char *name, const PluginFieldCollection *fc) override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(NMSPluginCreator);

}

#undef RETINANET_PLUGIN_NAME
#undef RETINANET_PLUGIN_VERSION
#undef RETINANET_PLUGIN_NAMESPACE