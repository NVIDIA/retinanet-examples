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

#include "decode_rotate.h"
#include "utils.h"

#include <algorithm>
#include <cstdint>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

namespace odtk {
namespace cuda {

int decode_rotate(int batch_size,
  const void *const *inputs, void *const *outputs,
  size_t height, size_t width, size_t scale,
  size_t num_anchors, size_t num_classes,
  const std::vector<float> &anchors, float score_thresh, int top_n,
  void *workspace, size_t workspace_size, cudaStream_t stream) {

  int scores_size = num_anchors * num_classes * height * width;

  if (!workspace || !workspace_size) {
    // Return required scratch space size cub style
    workspace_size  = get_size_aligned<float>(anchors.size()); // anchors
    workspace_size += get_size_aligned<bool>(scores_size);     // flags
    workspace_size += get_size_aligned<int>(scores_size);      // indices
    workspace_size += get_size_aligned<int>(scores_size);      // indices_sorted
    workspace_size += get_size_aligned<float>(scores_size);    // scores
    workspace_size += get_size_aligned<float>(scores_size);    // scores_sorted

    size_t temp_size_flag = 0;
    cub::DeviceSelect::Flagged((void *)nullptr, temp_size_flag,
      cub::CountingInputIterator<int>(scores_size),
      (bool *)nullptr, (int *)nullptr, (int *)nullptr, scores_size);
    size_t temp_size_sort = 0;
    cub::DeviceRadixSort::SortPairsDescending((void *)nullptr, temp_size_sort,
      (float *)nullptr, (float *)nullptr, (int *)nullptr, (int *)nullptr, scores_size);
    workspace_size += std::max(temp_size_flag, temp_size_sort);

    return workspace_size;
  }

  auto anchors_d = get_next_ptr<float>(anchors.size(), workspace, workspace_size);
  cudaMemcpyAsync(anchors_d, anchors.data(), anchors.size() * sizeof *anchors_d, cudaMemcpyHostToDevice, stream);

  auto on_stream = thrust::cuda::par.on(stream);

  auto flags = get_next_ptr<bool>(scores_size, workspace, workspace_size);
  auto indices = get_next_ptr<int>(scores_size, workspace, workspace_size);
  auto indices_sorted = get_next_ptr<int>(scores_size, workspace, workspace_size);
  auto scores = get_next_ptr<float>(scores_size, workspace, workspace_size);
  auto scores_sorted = get_next_ptr<float>(scores_size, workspace, workspace_size);

  for (int batch = 0; batch < batch_size; batch++) {
    auto in_scores = static_cast<const float *>(inputs[0]) + batch * scores_size;
    auto in_boxes = static_cast<const float *>(inputs[1]) + batch * (scores_size / num_classes) * 6; //From 4

    auto out_scores = static_cast<float *>(outputs[0]) + batch * top_n;
    auto out_boxes = static_cast<float6 *>(outputs[1]) + batch * top_n; // From float4
    auto out_classes = static_cast<float *>(outputs[2]) + batch * top_n;

    // Discard scores below threshold
    thrust::transform(on_stream, in_scores, in_scores + scores_size,
      flags, thrust::placeholders::_1 > score_thresh);

    int *num_selected = reinterpret_cast<int *>(indices_sorted);
    cub::DeviceSelect::Flagged(workspace, workspace_size, cub::CountingInputIterator<int>(0),
      flags, indices, num_selected, scores_size, stream);
    cudaStreamSynchronize(stream);
    int num_detections = *thrust::device_pointer_cast(num_selected);

    // Only keep top n scores
    auto indices_filtered = indices;
    if (num_detections > top_n) {
      thrust::gather(on_stream, indices, indices + num_detections,
        in_scores, scores);
      cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
        scores, scores_sorted, indices, indices_sorted, num_detections, 0, sizeof(*scores)*8, stream);
        indices_filtered = indices_sorted;
        num_detections = top_n;
    }

    // Gather boxes
    bool has_anchors = !anchors.empty();
    thrust::transform(on_stream, indices_filtered, indices_filtered + num_detections,
      thrust::make_zip_iterator(thrust::make_tuple(out_scores, out_boxes, out_classes)),
      [=] __device__ (int i) {
        int x = i % width;
        int y = (i / width) % height;
        int a = (i / num_classes / height / width) % num_anchors;
        int cls = (i / height / width) % num_classes;

        float6 box = make_float6(
          make_float4(
            in_boxes[((a * 6 + 0) * height + y) * width + x],
            in_boxes[((a * 6 + 1) * height + y) * width + x],
            in_boxes[((a * 6 + 2) * height + y) * width + x],
            in_boxes[((a * 6 + 3) * height + y) * width + x]
          ),
          make_float2(
            in_boxes[((a * 6 + 4) * height + y) * width + x],
            in_boxes[((a * 6 + 5) * height + y) * width + x]
          )
        );

        if (has_anchors) {
          // Add anchors offsets to deltas
          float x = (i % width) * scale;
          float y = ((i / width)  % height) * scale;
          float *d = anchors_d + 4*a;

          float x1 = x + d[0];
          float y1 = y + d[1];
          float x2 = x + d[2];
          float y2 = y + d[3];

          float w = x2 - x1 + 1.0f;
          float h = y2 - y1 + 1.0f;
          float pred_ctr_x = box.x1 * w + x1 + 0.5f * w;
          float pred_ctr_y = box.y1 * h + y1 + 0.5f * h;
          float pred_w = exp(box.x2) * w;
          float pred_h = exp(box.y2) * h;
          float pred_sin = box.s;
          float pred_cos = box.c;

          box = make_float6(
            make_float4(
              max(0.0f, pred_ctr_x - 0.5f * pred_w),
              max(0.0f, pred_ctr_y - 0.5f * pred_h),
              min(pred_ctr_x + 0.5f * pred_w - 1.0f, width * scale - 1.0f),
              min(pred_ctr_y + 0.5f * pred_h - 1.0f, height * scale - 1.0f)
            ),
            make_float2(pred_sin, pred_cos)
          );
        }

        return thrust::make_tuple(in_scores[i], box, cls);
      });

    // Zero-out unused scores
    if (num_detections < top_n) {
      thrust::fill(on_stream, out_scores + num_detections,
        out_scores + top_n, 0.0f);
      thrust::fill(on_stream, out_classes + num_detections,
        out_classes + top_n, 0.0f);
    }
  }

  return 0;
}

}
}
