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
#include <stdexcept>
#include <cstdint>
#include <thrust/functional.h>

#define CUDA_ALIGN 256

struct float6
{
  float x1, y1, x2, y2, s, c; 
};

inline __host__ __device__ float6 make_float6(float4 f, float2 t)
{
  float6 fs;
  fs.x1 = f.x; fs.y1 = f.y; fs.x2 = f.z; fs.y2 = f.w; fs.s = t.x; fs.c = t.y;
  return fs;
}

template <typename T>
inline size_t get_size_aligned(size_t num_elem) {
    size_t size = num_elem * sizeof(T);
    size_t extra_align = 0;
    if (size % CUDA_ALIGN != 0) {
        extra_align = CUDA_ALIGN - size % CUDA_ALIGN;
    }
    return size + extra_align;
}

template <typename T>
inline T *get_next_ptr(size_t num_elem, void *&workspace, size_t &workspace_size) {
  size_t size = get_size_aligned<T>(num_elem);
  if (size > workspace_size) {
    throw std::runtime_error("Workspace is too small!");
  }
  workspace_size -= size;
  T *ptr = reinterpret_cast<T *>(workspace);
  workspace = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(workspace) + size);
  return ptr;
}
