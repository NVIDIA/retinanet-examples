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

#include "nms_iou.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#include <thrust/system/cuda/detail/cub/iterator/counting_input_iterator.cuh>

constexpr int   kTPB     = 64;  // threads per block
constexpr int   kCorners = 4;
constexpr int   kPoints  = 8;

namespace retinanet {
namespace cuda {

class Vector {
public:
    __host__ __device__ Vector( );  // Default constructor
    __host__ __device__ ~Vector( );  // Deconstructor
    __host__ __device__ Vector( float2 const point );
    float2 const p;
    friend class Line;

private:
    __host__ __device__ float cross( Vector const v ) const;
};

Vector::Vector( ) : p( make_float2( 0.0f, 0.0f ) ) {}

Vector::~Vector( ) {}

Vector::Vector( float2 const point ) : p( point ) {}

float Vector::cross( Vector const v ) const {
    return ( p.x * v.p.y - p.y * v.p.x );
}

class Line {
public:
    __host__ __device__ Line( );  // Default constructor
    __host__ __device__ ~Line( );  // Deconstructor
    __host__ __device__ Line( Vector const v1, Vector const v2 );
    __host__ __device__ float call( Vector const v ) const;
    __host__ __device__ float2 intersection( Line const l ) const;

private:
    float const a;
    float const b;
    float const c;
};

Line::Line( ) : a( 0.0f ), b( 0.0f ), c( 0.0f ) {}

Line::~Line( ) {}

Line::Line( Vector const v1, Vector const v2 ) : a( v2.p.y - v1.p.y ), b( v1.p.x - v2.p.x ), c( v2.cross( v1 ) ) {}

float Line::call( Vector const v ) const {
    return ( a * v.p.x + b * v.p.y + c );
}

float2 Line::intersection( Line const l ) const {
    float w { a * l.b - b * l.a };
    return ( make_float2( ( b * l.c - c * l.b ) / w, ( c * l.a - a * l.c ) / w ) );
}

template<typename T>
__host__ __device__ void rotateLeft( T *array, int const &count ) {
    T temp = array[0];
    for ( int i = 0; i < count - 1; i++ )
        array[i] = array[i + 1];
    array[count - 1] = temp;
}

__host__ __device__ static __inline__ float2 padfloat2( float2 a, float2 b ) {
  float2 res;
  res.x = a.x + b.x;
  res.y = a.y + b.y;
  return res;
}

__device__ float IntersectionArea( float2 *mrect, float2 *mrect_shift, float2 *intersection ) {
    int count = kCorners;
    for ( int i = 0; i < kCorners; i++ ) {
        float2 intersection_shift[kPoints] {};
        for ( int k = 0; k < count; k++ )
            intersection_shift[k] = intersection[k];
        float line_values[kPoints] {};
        Vector const r1( mrect[i] );
        Vector const r2( mrect_shift[i] );
        Line const   line1( r1, r2 );
        for ( int j = 0; j < count; j++ ) {
            Vector const inter( intersection[j] );
            line_values[j] = line1.call( inter );
        }
        float line_values_shift[kPoints] {};

#pragma unroll
        for ( int k = 0; k < kPoints; k++ )
            line_values_shift[k] = line_values[k];
        rotateLeft( line_values_shift, count );
        rotateLeft( intersection_shift, count );
        float2 new_intersection[kPoints] {};
        int temp = count;
        count = 0;
        for ( int j = 0; j < temp; j++ ) {
            if ( line_values[j] <= 0 ) {
                new_intersection[count] = intersection[j];
                count++;
            }
            if ( ( line_values[j] * line_values_shift[j] ) <= 0 ) {
                Vector const r3( intersection[j] );
                Vector const r4( intersection_shift[j] );
                Line const Line( r3, r4 );
                new_intersection[count] = line1.intersection( Line );
                count++;
            }
        }
        for ( int k = 0; k < count; k++ )
            intersection[k] = new_intersection[k];
    }

    float2 intersection_shift[kPoints] {};

    for ( int k = 0; k < count; k++ )
        intersection_shift[k] = intersection[k];
    rotateLeft( intersection_shift, count );

    // Intersection
    float intersection_area = 0.0f;
    if ( count > 2 ) {
        for ( int k = 0; k < count; k++ )
            intersection_area +=
                intersection[k].x * intersection_shift[k].y - intersection[k].y * intersection_shift[k].x;
    }
    return ( abs( intersection_area / 2.0f ) );
}

__global__ void nms_rotate_kernel(const int num_per_thread, const float threshold, const int num_detections, 
  const int *indices, float *scores, const float *classes, const float6 *boxes ) {
  // Go through detections by descending score
  for ( int m = 0; m < num_detections; m++ ) {
    for ( int n = 0; n < num_per_thread; n++ ) {
      int ii = threadIdx.x * num_per_thread + n;
      if ( ii < num_detections && m < ii && scores[m] > 0.0f ) {
        int idx     = indices[ii];
        int max_idx = indices[m];
        int icls    = classes[idx];
        int mcls    = classes[max_idx];
        if ( mcls == icls ) {
          float6 ibox = make_float6( make_float4( boxes[idx].x1,
                                                  boxes[idx].y1,
                                                  boxes[idx].x2,
                                                  boxes[idx].y2 ),
                                  make_float2( boxes[idx].s, boxes[idx].c ) );
          float6 mbox = make_float6( make_float4( boxes[max_idx].x1,
                                                  boxes[max_idx].y1,
                                                  boxes[max_idx].x2,
                                                  boxes[max_idx].y2 ),
                                  make_float2( boxes[idx].s, boxes[idx].c ) );
          float2 intersection[kPoints] { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                      -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f };
          float2 irect[kPoints] {};
          float2 irect_shift[kPoints] {};
          float2 mrect[kPoints] {};
          float2 mrect_shift[kPoints] {};
          float2 icent = { ( ibox.x1 + ibox.x2 ) / 2.0f, ( ibox.y1 + ibox.y2 ) / 2.0f };
          float2 mcent = { ( mbox.x1 + mbox.x2 ) / 2.0f, ( mbox.y1 + mbox.y2 ) / 2.0f };
          float2 iboxc[kCorners] = { ibox.x1 - icent.x, ibox.y1 - icent.y, ibox.x2 - icent.x,
                                  ibox.y1 - icent.y, ibox.x2 - icent.x, ibox.y2 - icent.y,
                                  ibox.x1 - icent.x, ibox.y2 - icent.y };
          float2 mboxc[kCorners] = { mbox.x1 - mcent.x, mbox.y1 - mcent.y, mbox.x2 - mcent.x,
                                  mbox.y1 - mcent.y, mbox.x2 - mcent.x, mbox.y2 - mcent.y,
                                  mbox.x1 - mcent.x, mbox.y2 - mcent.y };
          float2 pad;
#pragma unroll
          for ( int b = 0; b < kCorners; b++ ) {
            if ( ( iboxc[b].x * ibox.c - iboxc[b].y * ibox.s ) + icent.x == ( mboxc[b].x * mbox.c - mboxc[b].y * mbox.s ) + mcent.x )
              pad.x = 0.001f;
            else
              pad.x = 0.0f;
            if ( ( iboxc[b].y * ibox.c + iboxc[b].x * ibox.s ) + icent.y == ( mboxc[b].y * mbox.c + mboxc[b].x * mbox.s ) + mcent.y )
              pad.y = 0.001f;
            else
              pad.y = 0.0f;
            intersection[b] = { ( iboxc[b].x * ibox.c - iboxc[b].y * ibox.s ) + icent.x + pad.x,
                                ( iboxc[b].y * ibox.c + iboxc[b].x * ibox.s ) + icent.y + pad.y};
            irect[b]        = { ( iboxc[b].x * ibox.c - iboxc[b].y * ibox.s ) + icent.x,
                        ( iboxc[b].y * ibox.c + iboxc[b].x * ibox.s ) + icent.y };
            irect_shift[b]  = { ( iboxc[b].x * ibox.c - iboxc[b].y * ibox.s ) + icent.x,
                            ( iboxc[b].y * ibox.c + iboxc[b].x * ibox.s ) + icent.y };
            mrect[b]        = { ( mboxc[b].x * mbox.c - mboxc[b].y * mbox.s ) + mcent.x,
                        ( mboxc[b].y * mbox.c + mboxc[b].x * mbox.s ) + mcent.y };
            mrect_shift[b]  = { ( mboxc[b].x * mbox.c - mboxc[b].y * mbox.s ) + mcent.x,
                            ( mboxc[b].y * mbox.c + mboxc[b].x * mbox.s ) + mcent.y };
          }
          rotateLeft( irect_shift, 4 );
          rotateLeft( mrect_shift, 4 );
          float intersection_area = IntersectionArea( mrect, mrect_shift, intersection );
          // Union
          float irect_area = 0.0f;
          float mrect_area = 0.0f;
#pragma unroll
          for ( int k = 0; k < kCorners; k++ ) {
            irect_area += irect[k].x * irect_shift[k].y - irect[k].y * irect_shift[k].x;
            mrect_area += mrect[k].x * mrect_shift[k].y - mrect[k].y * mrect_shift[k].x;
          }
          float union_area = ( abs( irect_area ) + abs( mrect_area ) ) / 2.0f;
          float overlap;
          if ( isnan( intersection_area ) && isnan( union_area ) ) {
            overlap = 1.0f;
          } else if ( isnan( intersection_area ) ) {
            overlap = 0.0f;
          } else {
            overlap = intersection_area / ( union_area - intersection_area );  // Check nans and inf
          }
          if ( overlap > threshold ) {
            scores[ii] = 0.0f;
          }
        }
      }
    }
    // Sync discarded detections
    __syncthreads( );
  }
}

int nms_rotate(int batch_size, const void *const *inputs, void **outputs, size_t count, 
  int detections_per_im, float nms_thresh, void *workspace, size_t workspace_size, cudaStream_t stream ) {
  if ( !workspace || !workspace_size ) {
    // Return required scratch space size cub style
    workspace_size = get_size_aligned<bool>( count );    // flags
    workspace_size += get_size_aligned<int>( count );    // indices
    workspace_size += get_size_aligned<int>( count );    // indices_sorted
    workspace_size += get_size_aligned<float>( count );  // scores
    workspace_size += get_size_aligned<float>( count );  // scores_sorted
    size_t temp_size_flag = 0;
    thrust::cuda_cub::cub::DeviceSelect::Flagged(( void * )nullptr,
                                                temp_size_flag,
                                                thrust::cuda_cub::cub::CountingInputIterator<int>( count ),
                                                ( bool * )nullptr,
                                                ( int * )nullptr,
                                                ( int * )nullptr,
                                                count );
    size_t temp_size_sort = 0;
    thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(( void * )nullptr,
                                                                temp_size_sort,
                                                                ( float * )nullptr,
                                                                ( float * )nullptr,
                                                                ( int * )nullptr,
                                                                ( int * )nullptr,
                                                                count );
    workspace_size += std::max( temp_size_flag, temp_size_sort );
    return workspace_size;
  }

  auto on_stream = thrust::cuda::par.on( stream );
  auto flags          = get_next_ptr<bool>( count, workspace, workspace_size );
  auto indices        = get_next_ptr<int>( count, workspace, workspace_size );
  auto indices_sorted = get_next_ptr<int>( count, workspace, workspace_size );
  auto scores         = get_next_ptr<float>( count, workspace, workspace_size );
  auto scores_sorted  = get_next_ptr<float>( count, workspace, workspace_size );
  for ( int batch = 0; batch < batch_size; batch++ ) {
    auto in_scores  = static_cast<const float *>( inputs[0] ) + batch * count;
    auto in_boxes   = static_cast<const float6 *>( inputs[1] ) + batch * count;
    auto in_classes = static_cast<const float *>( inputs[2] ) + batch * count;
    auto out_scores  = static_cast<float *>( outputs[0] ) + batch * detections_per_im;
    auto out_boxes   = static_cast<float6 *>( outputs[1] ) + batch * detections_per_im;
    auto out_classes = static_cast<float *>( outputs[2] ) + batch * detections_per_im;
    // Discard null scores
    thrust::transform( on_stream, in_scores, in_scores + count, flags, thrust::placeholders::_1 > 0.0f );
    int *num_selected = reinterpret_cast<int *>( indices_sorted );
    thrust::cuda_cub::cub::DeviceSelect::Flagged(workspace,
                                                workspace_size,
                                                thrust::cuda_cub::cub::CountingInputIterator<int>( 0 ),
                                                flags,
                                                indices,
                                                num_selected,
                                                count,
                                                stream );
    cudaStreamSynchronize( stream );
    int num_detections = *thrust::device_pointer_cast( num_selected );
    // Sort scores and corresponding indices
    thrust::gather( on_stream, indices, indices + num_detections, in_scores, scores );
    thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace,
                                                                workspace_size,
                                                                scores,
                                                                scores_sorted,
                                                                indices,
                                                                indices_sorted,
                                                                num_detections,
                                                                0,
                                                                sizeof( *scores ) * 8,
                                                                stream );  // From 8
    // Launch actual NMS kernel - 1 block with each thread handling n detections
    const int max_threads    = 1024;
    int       num_per_thread = ceil( ( float )num_detections / max_threads );
    nms_rotate_kernel<<<1, max_threads, 0, stream>>>(
        num_per_thread, nms_thresh, num_detections, indices_sorted, scores_sorted, in_classes, in_boxes );
    // Re-sort with updated scores
    thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace,
                                                                workspace_size,
                                                                scores_sorted,
                                                                scores,
                                                                indices_sorted,
                                                                indices,
                                                                num_detections,
                                                                0,
                                                                sizeof( *scores ) * 8,
                                                                stream );  // From 8
    // Gather filtered scores, boxes, classes
    num_detections = min( detections_per_im, num_detections );
    cudaMemcpyAsync( out_scores, scores, num_detections * sizeof *scores, cudaMemcpyDeviceToDevice, stream );
    if ( num_detections < detections_per_im ) {
      thrust::fill_n( on_stream, out_scores + num_detections, detections_per_im - num_detections, 0 );
    }
    thrust::gather( on_stream, indices, indices + num_detections, in_boxes, out_boxes );
    thrust::gather( on_stream, indices, indices + num_detections, in_classes, out_classes );
  }
  return 0;
}

__global__ void iou_cuda_kernel(int const numBoxes, int const numAnchors,
  float2 const *b_box_vals, float2 const *a_box_vals, float *iou_vals ) {
  int t      = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int combos = numBoxes * numAnchors;
  for ( int tid = t; tid < combos; tid += stride ) {
    float2 intersection[kPoints] { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f };
    float2 rect1[kPoints] {};
    float2 rect1_shift[kPoints] {};
    float2 rect2[kPoints] {};
    float2 rect2_shift[kPoints] {};
    float2 pad;
#pragma unroll
    for ( int b = 0; b < kCorners; b++ ) {
      if ( b_box_vals[( static_cast<int>( tid / numAnchors ) * kCorners + b )].x == a_box_vals[( tid * kCorners + b ) % ( numAnchors * kCorners )].x )
        pad.x = 0.001f;
      else
        pad.x = 0.0f;
      if ( b_box_vals[( static_cast<int>( tid / numAnchors ) * kCorners + b )].y == a_box_vals[( tid * kCorners + b ) % ( numAnchors * kCorners )].y )
        pad.y = 0.001f;
      else
        pad.y = 0.0f;
      intersection[b] = padfloat2( b_box_vals[( static_cast<int>( tid / numAnchors ) * kCorners + b )], pad );
      rect1[b]        = b_box_vals[( static_cast<int>( tid / numAnchors ) * kCorners + b )];
      rect1_shift[b]  = b_box_vals[( static_cast<int>( tid / numAnchors ) * kCorners + b )];
      rect2[b]        = a_box_vals[( tid * kCorners + b ) % ( numAnchors * kCorners )];
      rect2_shift[b]  = a_box_vals[( tid * kCorners + b ) % ( numAnchors * kCorners )];
    }
    rotateLeft( rect1_shift, 4 );
    rotateLeft( rect2_shift, 4 );
    float intersection_area = IntersectionArea( rect2, rect2_shift, intersection );
    // Union
    float rect1_area = 0.0f;
    float rect2_area = 0.0f;
#pragma unroll
    for ( int k = 0; k < kCorners; k++ ) {
        rect1_area += rect1[k].x * rect1_shift[k].y - rect1[k].y * rect1_shift[k].x;
        rect2_area += rect2[k].x * rect2_shift[k].y - rect2[k].y * rect2_shift[k].x;
    }
    float union_area = ( abs( rect1_area ) + abs( rect2_area ) ) / 2.0f;
    float iou_val = intersection_area / ( union_area - intersection_area );
    // Write out answer
    if ( isnan( intersection_area ) && isnan( union_area ) ) {
        iou_vals[tid] = 1.0f;
    } else if ( isnan( intersection_area ) ) {
        iou_vals[tid] = 0.0f;
    } else {
        iou_vals[tid] = iou_val;
    }
  }
}

int iou( const void *const *inputs, void **outputs, int num_boxes, int num_anchors, cudaStream_t stream ) {
  auto boxes    = static_cast<const float2 *>( inputs[0] );
  auto anchors  = static_cast<const float2 *>( inputs[1] );
  auto iou_vals = static_cast<float *>( outputs[0] );
  int numSMs;
  cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 );
  int threadsPerBlock = kTPB;
  int blocksPerGrid   = numSMs * 10;
  iou_cuda_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>( num_anchors, num_boxes, anchors, boxes, iou_vals );
  return 0;
}

}
}
