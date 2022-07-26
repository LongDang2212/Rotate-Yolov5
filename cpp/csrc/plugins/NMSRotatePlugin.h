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

#include "../cuda/nms_iou.h"

using namespace nvinfer1;

#define PLUGIN_NAME "RotateNMS"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""



namespace ryolo {

  class NMSRotatePlugin : public IPluginV2DynamicExt
  {
    float _nms_thresh;
    int _detections_per_im;

    size_t _count;
    mutable int size = -1;

  protected:
    void deserialize(void const *data, size_t length)
    {
      const char *d = static_cast<const char *>(data);
      read(d, _nms_thresh);
      read(d, _detections_per_im);
      read(d, _count);
    }

    size_t getSerializationSize() const noexcept override 
    {
      return sizeof(_nms_thresh) + sizeof(_detections_per_im) + sizeof(_count);
    }

    void serialize(void *buffer) const noexcept override
    {
      char *d = static_cast<char *>(buffer);
      write(d, _nms_thresh);
      write(d, _detections_per_im);
      write(d, _count);
    }

  public:
    NMSRotatePlugin(float nms_thresh, int detections_per_im)
        : _nms_thresh(nms_thresh), _detections_per_im(detections_per_im)
    {
      assert(nms_thresh > 0);
      assert(detections_per_im > 0);
    }

    NMSRotatePlugin(float nms_thresh, int detections_per_im, size_t count)
        : _nms_thresh(nms_thresh), _detections_per_im(detections_per_im), _count(count)
    {
      assert(nms_thresh > 0);
      assert(detections_per_im > 0);
      assert(count > 0);
    }

    NMSRotatePlugin(void const *data, size_t length)
    {
      this->deserialize(data, length);
    }

    const char *getPluginType() const noexcept override
    {
      return PLUGIN_NAME;
    }

    const char *getPluginVersion() const noexcept override
    {
      return PLUGIN_VERSION;
    }

    int getNbOutputs() const noexcept override
    {
      return 3;
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs,
                                  int nbInputs, IExprBuilder &exprBuilder) noexcept override
    {
      DimsExprs output(inputs[0]);
      if (outputIndex == 1)
      {
        output.d[1] = exprBuilder.constant(_detections_per_im * 6);
      }
      else
      {
        output.d[1] = exprBuilder.constant(_detections_per_im);
      }
      output.d[2] = exprBuilder.constant(1);
      output.d[3] = exprBuilder.constant(1);
      return output;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut,
                                   int nbInputs, int nbOutputs) noexcept override
    {
      assert(nbInputs == 3);
      assert(nbOutputs == 3);
      assert(pos < 6); // 2+2
      return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR;
    }

    int initialize()noexcept override { return 0; }

    void terminate()noexcept override {}

    size_t getWorkspaceSize(const PluginTensorDesc *inputs,
                            int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const noexcept override
    {
      if (size < 0)
      {
        size = cuda::nms_rotate(inputs->dims.d[0], nullptr, nullptr, _count,
                                 _detections_per_im, _nms_thresh,
                                 nullptr, 0, nullptr);
      }
      return size;
    }

    int enqueue(const PluginTensorDesc *inputDesc,
                const PluginTensorDesc *outputDesc, const void *const *inputs,
                void *const *outputs, void *workspace, cudaStream_t stream)noexcept
    {
      return cuda::nms_rotate(inputDesc->dims.d[0], inputs, outputs, _count,
                               _detections_per_im, _nms_thresh,
                               workspace, getWorkspaceSize(inputDesc, 3, outputDesc, 3), stream);
    }

    void destroy()noexcept override
    {
      delete this;
    }

    const char *getPluginNamespace() const noexcept override
    {
      return PLUGIN_NAMESPACE;
    }

    void setPluginNamespace(const char *N)noexcept override {}

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept
    {
      assert(index < 3);
      return DataType::kFLOAT;
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs,
                         const DynamicPluginTensorDesc *out, int nbOutputs)noexcept
    {
      assert(nbInputs == 3);
      assert(in[0].desc.dims.d[1] == in[2].desc.dims.d[1]);
      assert(in[1].desc.dims.d[1] == in[2].desc.dims.d[1] * 6);
      _count = in[0].desc.dims.d[1];
    }

    IPluginV2DynamicExt *clone() const noexcept
    {
      return new NMSRotatePlugin(_nms_thresh, _detections_per_im, _count);
    }

  private:
    template <typename T>
    void write(char *&buffer, const T &val) const
    {
      *reinterpret_cast<T *>(buffer) = val;
      buffer += sizeof(T);
    }

    template <typename T>
    void read(const char *&buffer, T &val)
    {
      val = *reinterpret_cast<const T *>(buffer);
      buffer += sizeof(T);
    }
  };

  class NMSRotatePluginCreator : public IPluginCreator
  {
  public:
    NMSRotatePluginCreator() {}

    const char *getPluginNamespace() const noexcept override
    {
      return PLUGIN_NAMESPACE;
    }
    const char *getPluginName() const noexcept override
    {
      return PLUGIN_NAME;
    }

    const char *getPluginVersion() const noexcept override
    {
      return PLUGIN_VERSION;
    }

    //Was IPluginV2
    IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength)noexcept override
    {
      return new NMSRotatePlugin(serialData, serialLength);
    }

    //Was IPluginV2
    void setPluginNamespace(const char *N)noexcept override {}
    const PluginFieldCollection *getFieldNames()noexcept override { return nullptr; }
    IPluginV2DynamicExt *createPlugin(const char *name, const PluginFieldCollection *fc)noexcept override { return nullptr; }
  };

  REGISTER_TENSORRT_PLUGIN(NMSRotatePluginCreator);

}

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
