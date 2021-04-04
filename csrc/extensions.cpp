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

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <cmath>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include <optional>

#include "engine.h"
#include "cuda/decode.h"
#include "cuda/decode_rotate.h"
#include "cuda/nms.h"
#include "cuda/nms_iou.h"
#include <stdio.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


vector<at::Tensor> iou(at::Tensor boxes, at::Tensor anchors) {

    CHECK_INPUT(boxes);
    CHECK_INPUT(anchors);

    int num_boxes = boxes.numel() / 8;
    int num_anchors = anchors.numel() / 8;
    auto options = boxes.options();

    auto iou_vals = at::zeros({num_boxes*num_anchors}, options);

    // Calculate Polygon IOU
    vector<void *> inputs = {boxes.data_ptr(), anchors.data_ptr()};
    vector<void *> outputs = {iou_vals.data_ptr()};

    odtk::cuda::iou(inputs.data(), outputs.data(), num_boxes, num_anchors, at::cuda::getCurrentCUDAStream());

    auto shape = std::vector<int64_t>{num_anchors, num_boxes};

    return {iou_vals.reshape(shape)};
}

vector<at::Tensor> decode(at::Tensor cls_head, at::Tensor box_head,
        vector<float> &anchors, int scale, float score_thresh, int top_n, bool rotated=false) {

    CHECK_INPUT(cls_head);
    CHECK_INPUT(box_head);

    int num_boxes = (!rotated) ? 4 : 6;
    int batch = cls_head.size(0);
    int num_anchors = anchors.size() / 4;
    int num_classes = cls_head.size(1) / num_anchors;
    int height = cls_head.size(2);
    int width = cls_head.size(3);
    auto options = cls_head.options();

    auto scores = at::zeros({batch, top_n}, options);
    auto boxes = at::zeros({batch, top_n, num_boxes}, options);
    auto classes = at::zeros({batch, top_n}, options);

    vector<void *> inputs = {cls_head.data_ptr(), box_head.data_ptr()};
    vector<void *> outputs = {scores.data_ptr(), boxes.data_ptr(), classes.data_ptr()};

    if(!rotated) {
        // Create scratch buffer
        int size = odtk::cuda::decode(batch, nullptr, nullptr, height, width, scale,
            num_anchors, num_classes, anchors, score_thresh, top_n, nullptr, 0, nullptr);
        auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

        // Decode boxes
        odtk::cuda::decode(batch, inputs.data(), outputs.data(), height, width, scale,
            num_anchors, num_classes, anchors, score_thresh, top_n,
            scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());
    
    }
    else {
        // Create scratch buffer
        int size = odtk::cuda::decode_rotate(batch, nullptr, nullptr, height, width, scale,
            num_anchors, num_classes, anchors, score_thresh, top_n, nullptr, 0, nullptr);
        auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

        // Decode boxes
        odtk::cuda::decode_rotate(batch, inputs.data(), outputs.data(), height, width, scale,
            num_anchors, num_classes, anchors, score_thresh, top_n,
            scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());
    }
    
    return {scores, boxes, classes};
}

vector<at::Tensor> nms(at::Tensor scores, at::Tensor boxes, at::Tensor classes,
        float nms_thresh, int detections_per_im, bool rotated=false) {

    CHECK_INPUT(scores);
    CHECK_INPUT(boxes);
    CHECK_INPUT(classes);

    int num_boxes = (!rotated) ? 4 : 6;
    int batch = scores.size(0);
    int count = scores.size(1);
    auto options = scores.options();
    auto nms_scores = at::zeros({batch, detections_per_im}, scores.options());
    auto nms_boxes = at::zeros({batch, detections_per_im, num_boxes}, boxes.options());
    auto nms_classes = at::zeros({batch, detections_per_im}, classes.options());

    vector<void *> inputs = {scores.data_ptr(), boxes.data_ptr(), classes.data_ptr()};
    vector<void *> outputs = {nms_scores.data_ptr(), nms_boxes.data_ptr(), nms_classes.data_ptr()};

    if(!rotated) {
        // Create scratch buffer
        int size = odtk::cuda::nms(batch, nullptr, nullptr, count,
            detections_per_im, nms_thresh, nullptr, 0, nullptr);
        auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

        // Perform NMS
        odtk::cuda::nms(batch, inputs.data(), outputs.data(), count, detections_per_im, 
            nms_thresh, scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());
    }
    else {
        // Create scratch buffer
        int size = odtk::cuda::nms_rotate(batch, nullptr, nullptr, count,
            detections_per_im, nms_thresh, nullptr, 0, nullptr);
        auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

        // Perform NMS
        odtk::cuda::nms_rotate(batch, inputs.data(), outputs.data(), count,
            detections_per_im, nms_thresh, scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());
    }
    

    return {nms_scores, nms_boxes, nms_classes};
}

vector<at::Tensor> infer(odtk::Engine &engine, at::Tensor data, bool rotated=false) {
    CHECK_INPUT(data);

    int num_boxes = (!rotated) ? 4 : 6;
    int batch = data.size(0);
    auto input_size = engine.getInputSize();
    data = at::constant_pad_nd(data, {0, input_size[1] - data.size(3), 0, input_size[0] - data.size(2)});

    int num_detections = engine.getMaxDetections();
    auto scores = at::zeros({batch, num_detections}, data.options());
    auto boxes = at::zeros({batch, num_detections, num_boxes}, data.options());
    auto classes = at::zeros({batch, num_detections}, data.options());

    vector<void *> buffers;
    for (auto buffer : {data, scores, boxes, classes}) {
        buffers.push_back(buffer.data<float>());
    }

    engine.infer(buffers, batch);

    return {scores, boxes, classes};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<odtk::Engine>(m, "Engine")
        .def(pybind11::init<const char *, size_t, const vector<int>&, string, float, int,
            const vector<vector<float>>&, bool, float, int, const vector<string>&, string, string, bool>())
        .def("save", &odtk::Engine::save)
        .def("infer", &odtk::Engine::infer)
        .def_property_readonly("stride", &odtk::Engine::getStride)
        .def_property_readonly("input_size", &odtk::Engine::getInputSize)
        .def_static("load", [](const string &path) {
            return new odtk::Engine(path);
        })
        .def("__call__", [](odtk::Engine &engine, at::Tensor data, bool rotated=false) {
            return infer(engine, data, rotated);
        });
    m.def("decode", &decode);
    m.def("nms", &nms);
    m.def("iou", &iou);
}