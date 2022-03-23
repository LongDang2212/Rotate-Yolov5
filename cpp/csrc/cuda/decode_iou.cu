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

 #include "decode_iou.h"
 #include "utils.h"
 #include <cstdio>
 
 #include <algorithm>
 #include <cstdint>
 
 #include <thrust/device_ptr.h>
 #include <thrust/sequence.h>
 #include <thrust/execution_policy.h>
 #include <thrust/gather.h>
 #include <thrust/tabulate.h>
 #include <thrust/count.h>
 #include <thrust/find.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#include <thrust/system/cuda/detail/cub/iterator/counting_input_iterator.cuh>
 
 namespace ryolo
 {
     namespace cuda
     {
 
         __global__ void softmax_kernel(const float *data, float *scores, float *conf, float *boxes, int num_elem)
         {
             int idx = threadIdx.x + blockIdx.x * blockDim.x;
             if (idx >= num_elem)
                 return;
             
 
             for (int k = 0; k < 5; ++k)
             { // 5 == num_anchor_per_point
 
             
                 boxes[idx + num_elem * (6 * k + 0)] = 1 / (1.0f + expf(data[idx + num_elem * (8 * k + 0)] * -1.0));
                 boxes[idx + num_elem * (6 * k + 1)] = 1 / (1.0f + expf(data[idx + num_elem * (8 * k + 1)] * -1.0));
                 boxes[idx + num_elem * (6 * k + 2)] = 1 / (1.0f + expf(data[idx + num_elem * (8 * k + 2)] * -1.0));
                 boxes[idx + num_elem * (6 * k + 3)] = 1 / (1.0f + expf(data[idx + num_elem * (8 * k + 3)] * -1.0));
                 boxes[idx + num_elem * (6 * k + 4)] = 1 / (1.0f + expf(data[idx + num_elem * (8 * k + 4)] * -1.0));
                 float temp = expf(2 * data[idx + num_elem * (8 * k + 5)]);
                 boxes[idx + num_elem * (6 * k + 5)] = (temp - 1) / (temp + 1);
                 
                 // for(int i =0; i<6; i++)
                 // {
                 //     printf("%f  ",boxes[idx + num_elem * (6 * k + i)]);
                 // }
                 // printf("\n");
                 float score = data[idx + num_elem * (8 * k + 6)];
                 // printf("\n%f", score);
                 score = 1 / (1.0f + expf(score * -1.0));
 
                 scores[idx + num_elem * 1 * k] = score;
                 float cls = data[idx + num_elem * (8 * k + 7)];
                 cls = 1 / (1.0f + expf(cls * -1.0));
                 cls = cls * score;
                 conf[idx + num_elem * 1 * k] = cls;
             }
         }
 
         int decode(int batch_size,
                    const void *const *inputs, void *const *outputs,
                    size_t num_anchors, const std::vector<float> &anchors,
                    int top_n, size_t f_size, float score_thresh,
                    int stride,
                    void *workspace, size_t workspace_size, cudaStream_t stream)
         {
 
             if (!workspace || !workspace_size)
             {
                 // scratch space size cub style
                 workspace_size = get_size_aligned<float>(anchors.size());   // anchors
                 workspace_size += get_size_aligned<bool>(num_anchors);      // flags
                 workspace_size += get_size_aligned<int>(num_anchors);       // indices
                 workspace_size += get_size_aligned<int>(num_anchors);       // indices_sorted
                 workspace_size += get_size_aligned<float>(num_anchors);     // scores
                 workspace_size += get_size_aligned<float>(num_anchors);     // scores_sorted
                 workspace_size += get_size_aligned<float>(num_anchors);     // scores_softmax
                 workspace_size += get_size_aligned<float>(num_anchors);     // conf
                 workspace_size += get_size_aligned<float>(num_anchors * 6); // in_boxes
                workspace_size += get_size_aligned<float>(num_anchors);     // classes
 
                 size_t temp_size_flag = 0;
                 thrust::cuda_cub::cub::DeviceSelect::Flagged((void *)nullptr, temp_size_flag,
                                            thrust::cuda_cub::cub::CountingInputIterator<int>(num_anchors),
                                            (bool *)nullptr, (int *)nullptr, (int *)nullptr, num_anchors);
                 size_t temp_size_sort = 0;
                 thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending((void *)nullptr, temp_size_sort,
                                                           (float *)nullptr, (float *)nullptr, (int *)nullptr, (int *)nullptr, num_anchors);
                 workspace_size += std::max(temp_size_flag, temp_size_sort);
 
                 return workspace_size;
             }
 
             auto anchors_d = get_next_ptr<float>(anchors.size(), workspace, workspace_size);
             cudaMemcpyAsync(anchors_d, anchors.data(), anchors.size() * sizeof *anchors_d, cudaMemcpyHostToDevice, stream);
 
             auto on_stream = thrust::cuda::par.on(stream);
 
             auto flags = get_next_ptr<bool>(num_anchors, workspace, workspace_size);  // used for filtering flags by threshold
             auto indices = get_next_ptr<int>(num_anchors, workspace, workspace_size); // used for filtering index by threshold
             auto indices_sorted = get_next_ptr<int>(num_anchors, workspace, workspace_size);
             auto scores = get_next_ptr<float>(num_anchors, workspace, workspace_size);
             auto scores_sorted = get_next_ptr<float>(num_anchors, workspace, workspace_size);
             auto scores_softmax = get_next_ptr<float>(num_anchors, workspace, workspace_size);
             auto conf = get_next_ptr<float>(num_anchors, workspace, workspace_size);
 
             auto in_boxes = get_next_ptr<float>(num_anchors * 6, workspace, workspace_size);
 
             int thread_count;
             int num_anchor = 5;
 
             for (int batch = 0; batch < batch_size; batch++)
             {
                 auto in_data = static_cast<const float *>(inputs[0]) + batch * num_anchors * 8; // cx,cy,w,h,cos,sin,score,cls
                 auto out_scores = static_cast<float *>(outputs[0]) + batch * top_n;
                 auto out_boxes = static_cast<float6 *>(outputs[1]) + batch * top_n;
                 auto out_classes = static_cast<float *>(outputs[2]) + batch * top_n;
 
                 // sigmoid activation
                 const int thread_count_ = 512;
                 int num_elem = f_size * f_size;
                 thread_count = (num_elem < thread_count_) ? num_elem : thread_count_;
                 softmax_kernel<<<(num_elem + thread_count - 1) / thread_count, thread_count, 0, stream>>>(in_data, scores_softmax, conf, in_boxes, num_elem);
 
                 // Discard scores below threshold
                 thrust::transform(on_stream, scores_softmax, scores_softmax + num_anchors, flags, thrust::placeholders::_1 > score_thresh);
 
                 int *num_selected = reinterpret_cast<int *>(indices_sorted);
                 thrust::cuda_cub::cub::DeviceSelect::Flagged(workspace, workspace_size,
                                            thrust::cuda_cub::cub::CountingInputIterator<int>(0),
                                            flags, indices, num_selected, num_anchors, stream);
                 cudaStreamSynchronize(stream);
                 int num_detections = *thrust::device_pointer_cast(num_selected);
 
                 // Only keep top n scores
                 auto indices_filtered = indices;
                 if (num_detections > top_n)
                 {
                     // lấy score theo indices đã chọn ở trên, sort index theo score, đẩy vào scores
                     thrust::gather(on_stream, indices, indices + num_detections, scores_softmax, scores);
                     // sort các giá trị trong scores đẩy vào scores_sorted để lấy n giá trị
                     thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
                                                               scores, scores_sorted, indices, indices_sorted, num_detections, 0, sizeof(*scores) * 8, stream);
                     indices_filtered = indices_sorted;
                     num_detections = top_n;
                 }
 
                 // Gather boxes
                 // bool has_anchors = !anchors.empty();
                 thrust::transform(on_stream, indices_filtered, indices_filtered + num_detections,
                                   thrust::make_zip_iterator(thrust::make_tuple(out_scores, out_boxes, out_classes)),
                                   [=] __device__(int i)
                                   {
                                       int x = i % f_size;
                                       int y = (i / f_size) % f_size;
                                       int a = (i / f_size / f_size) % num_anchor;
 
                                       float cx = in_boxes[((a * 6 + 0) * f_size + y) * f_size + x];
                                       float cy = in_boxes[((a * 6 + 1) * f_size + y) * f_size + x];
                                       float w = in_boxes[((a * 6 + 2) * f_size + y) * f_size + x];
                                       float h = in_boxes[((a * 6 + 3) * f_size + y) * f_size + x];
                                       float cos = in_boxes[((a * 6 + 4) * f_size + y) * f_size + x];
                                       float sin = in_boxes[((a * 6 + 5) * f_size + y) * f_size + x];
 
                                       cx = (cx * 2.0f - 0.5f + x) * stride;
                                       cy = (cy * 2.0f - 0.5f + y) * stride;
                                       w = w * w * 4 * anchors_d[a * 2 + 0];
                                       h = h * h * 4 * anchors_d[a * 2 + 1];                                       
                                       
                                     //   convert to xyxyxyxy
                                       float x0 = -w / 2.0f;
                                       float x1 = w / 2.0f;
                                       float y0 = -h / 2.0f;
                                       float y1 = h / 2.0f;
 
                                       float xyxyxyxy[4][2] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
                                       float R[2][2] = {{cos, sin}, {sin, cos}};
                                       float temp[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
                                       for (int m = 0; m < 4; ++m)
                                           for (int j = 0; j < 2; ++j)
                                           {
                                               for (int k = 0; k < 2; ++k)
                                               {
                                                   temp[m][j] += xyxyxyxy[m][k] * R[k][j];
                                               }
                                           }
                                       for (int m = 0; m < 4; ++m)
                                       {
                                           temp[m][0] += cx;
                                           temp[m][1] += cy;
                                       }
                                       float x_min = temp[0][0];
                                       float x_max = temp[0][0];
                                       for(int m=1; m<4; m++)
                                       {
                                         if (temp[m][0]<x_min)
                                         x_min = temp[m][0];
                                         if (temp[m][0]>x_max)
                                         x_max = temp[m][0];
                                       }
                                       float y_min = temp[0][1];
                                       float y_max = temp[0][1];
                                       for(int m=1; m<4; m++)
                                       {
                                         if (temp[m][1]<y_min)
                                         y_min = temp[m][1];
                                         if (temp[m][1]>y_max)
                                         y_max = temp[m][1];
                                       }
                                       float6 box = make_float6(make_float4(x_min, y_min, x_max,y_max),make_float2(sin, cos));
                               
                                       return thrust::make_tuple(conf[i], box, 0);
                                   });
 
                 // Zero-out unused scores
                 if (num_detections < top_n)
                 {
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
