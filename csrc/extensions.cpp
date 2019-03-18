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

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include <optional>

#include "engine.h"
#include "cuda/decode.h"
#include "cuda/nms.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

vector<at::Tensor> decode(at::Tensor cls_head, at::Tensor box_head,
        vector<float> &anchors, int scale, float score_thresh, int top_n) {

    CHECK_INPUT(cls_head);
    CHECK_INPUT(box_head);

    int batch = cls_head.size(0);
    int num_anchors = anchors.size() / 4;
    int num_classes = cls_head.size(1) / num_anchors;
    int height = cls_head.size(2);
    int width = cls_head.size(3);
    auto options = cls_head.options();

    auto scores = at::zeros({batch, top_n}, options);
    auto boxes = at::zeros({batch, top_n, 4}, options);
    auto classes = at::zeros({batch, top_n}, options);


    // Create scratch buffer
    int size = retinanet::cuda::decode(batch, nullptr, nullptr, height, width, scale,
        num_anchors, num_classes, anchors, score_thresh, top_n, nullptr, 0, nullptr);
    auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

    // Decode boxes
    vector<void *> inputs = {cls_head.data_ptr(), box_head.data_ptr()};
    vector<void *> outputs = {scores.data_ptr(), boxes.data_ptr(), classes.data_ptr()};
    retinanet::cuda::decode(batch, inputs.data(), outputs.data(), height, width, scale,
        num_anchors, num_classes, anchors, score_thresh, top_n,
        scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());

    return {scores, boxes, classes};
}

vector<at::Tensor> nms(at::Tensor scores, at::Tensor boxes, at::Tensor classes,
        float nms_thresh, int detections_per_im) {

    CHECK_INPUT(scores);
    CHECK_INPUT(boxes);
    CHECK_INPUT(classes);

    int batch = scores.size(0);
    int count = scores.size(1);
    auto options = scores.options();

    auto nms_scores = at::zeros({batch, detections_per_im}, scores.options());
    auto nms_boxes = at::zeros({batch, detections_per_im, 4}, boxes.options());
    auto nms_classes = at::zeros({batch, detections_per_im}, classes.options());

    // Create scratch buffer
    int size = retinanet::cuda::nms(batch, nullptr, nullptr, count, 
        detections_per_im, nms_thresh, nullptr, 0, nullptr);
    auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

    // Perform NMS
    vector<void *> inputs = {scores.data_ptr(), boxes.data_ptr(), classes.data_ptr()};
    vector<void *> outputs = {nms_scores.data_ptr(), nms_boxes.data_ptr(), nms_classes.data_ptr()};
    retinanet::cuda::nms(batch, inputs.data(), outputs.data(), count,
        detections_per_im, nms_thresh,
        scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());

    return {nms_scores, nms_boxes, nms_classes};
}

vector<at::Tensor> infer(retinanet::Engine &engine, at::Tensor data) {
    CHECK_INPUT(data);
    
    int batch = data.size(0);
    auto input_size = engine.getInputSize();
    data = at::constant_pad_nd(data, {0, input_size[1] - data.size(3), 0, input_size[0] - data.size(2)});

    int num_detections = engine.getMaxDetections();
    auto scores = at::zeros({batch, num_detections}, data.options());
    auto boxes = at::zeros({batch, num_detections, 4}, data.options());
    auto classes = at::zeros({batch, num_detections}, data.options());

    vector<void *> buffers;
    for (auto buffer : {data, scores, boxes, classes}) {
        buffers.push_back(buffer.data<float>());
    }

    engine.infer(buffers, batch);

    return {scores, boxes, classes};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<retinanet::Engine>(m, "Engine")
        .def(pybind11::init<const char *, size_t, size_t, string, float,
            int, const vector<vector<float>>&, float, int, const vector<string>&, string, string, bool>())
        .def("save", &retinanet::Engine::save)
        .def("infer", &retinanet::Engine::infer)
        .def_property_readonly("stride", &retinanet::Engine::getStride)
        .def_property_readonly("input_size", &retinanet::Engine::getInputSize)
        .def_static("load", [](const string &path) {
            return new retinanet::Engine(path);
        })
        .def("__call__", [](retinanet::Engine &engine, at::Tensor data) { 
            return infer(engine, data);
        });
    m.def("decode", &decode);
    m.def("nms", &nms);
}
