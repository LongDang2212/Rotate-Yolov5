#pragma once

#include <NvInfer.h>

#include <cassert>
#include <vector>

#include "../cuda/decode_iou.h"

using namespace nvinfer1;

#define PLUGIN_NAME "RotateYolov5Decode"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace ryolo
{

  class DecodePlugin : public IPluginV2DynamicExt
  {
    float _score_thresh;
    int _top_n;
    std::vector<float> _anchors;
    int _stride;

    size_t _f_size;
    size_t _num_anchors;
    size_t _num_classes;
    mutable int size = -1;

  protected:
    void deserialize(void const *data, size_t length)
    {
      const char *d = static_cast<const char *>(data);
      read(d, _score_thresh);
      read(d, _top_n);
      size_t anchors_size;
      read(d, anchors_size);
      while (anchors_size--)
      {
        float val;
        read(d, val);
        _anchors.push_back(val);
      }
      read(d, _stride);
      read(d, _f_size);
      read(d, _num_anchors);
      read(d, _num_classes);
    }

    size_t getSerializationSize() const noexcept override
    {
      return sizeof(_score_thresh) + sizeof(_top_n) + sizeof(size_t) + sizeof(_stride) + sizeof(_f_size) + sizeof(_num_anchors)+sizeof(float) * _anchors.size()+sizeof(_num_classes);
    }

    void serialize(void *buffer)const noexcept override
    {
      char *d = static_cast<char *>(buffer);
      write(d, _score_thresh);
      write(d, _top_n);
      write(d, _anchors.size());
      for (auto &val : _anchors)
      {
        write(d, val);
      }
      write(d, _stride);
      write(d, _f_size);
      write(d, _num_anchors);
      write(d, _num_classes);
    }

  public:
    DecodePlugin(float score_thresh, int top_n, int stride, std::vector<float> const &anchors)
        : _score_thresh(score_thresh), _top_n(top_n),_anchors(anchors),
          _stride(stride) {}

    DecodePlugin(float score_thresh, int top_n, int stride,
                 size_t f_size, size_t num_anchors, const std::vector<float> &anchors)
        : _score_thresh(score_thresh), _top_n(top_n), _stride(stride), _f_size(f_size), _num_anchors(num_anchors), _anchors(anchors) {}

    // Sử dụng khi load engine
    DecodePlugin(void const *data, size_t length)
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
        output.d[1] = exprBuilder.constant(_top_n * 6);
      }
      else
      {
        output.d[1] = exprBuilder.constant(_top_n);
      }
      output.d[2] = exprBuilder.constant(1);
      output.d[3] = exprBuilder.constant(1);

      return output;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut,
                                   int nbInputs, int nbOutputs)noexcept override
    {
      assert(nbInputs == 1);
      assert(nbOutputs == 3);
      assert(pos < 5);
      return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR;
    }

    int initialize()noexcept override { return 0; }

    void terminate()noexcept override {}

    size_t getWorkspaceSize(const PluginTensorDesc *inputs,
                            int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const noexcept override
    {
      if (size < 0)
      {
        size = cuda::decode(inputs->dims.d[0], nullptr, nullptr,
                            _num_anchors, _anchors,_top_n, _f_size, _score_thresh, _stride,
                            nullptr, 0, nullptr);
      }
      return size;
    }

    int enqueue(const PluginTensorDesc *inputDesc,
                const PluginTensorDesc *outputDesc, const void *const *inputs,
                void *const *outputs, void *workspace, cudaStream_t stream) noexcept
    {

      return cuda::decode(inputDesc->dims.d[0], inputs, outputs,
                          _num_anchors,_anchors, _top_n, _f_size, _score_thresh, _stride,
                          workspace, getWorkspaceSize(inputDesc, 1, outputDesc, 3), stream);
    }

    void destroy()noexcept override
    {
      delete this;
    };

    const char *getPluginNamespace() const noexcept override
    {
      return PLUGIN_NAMESPACE;
    }

    void setPluginNamespace(const char *N) noexcept override {}

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept
    {
      assert(index < 3);
      return DataType::kFLOAT;
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs,
                         const DynamicPluginTensorDesc *out, int nbOutputs)noexcept
    {
      assert(nbInputs == 1);
      assert(nbOutputs == 3);
      
      auto const &data_dims = in[0].desc.dims;
      _f_size = data_dims.d[2];
      _num_anchors = _f_size * _f_size * 5;
      _num_classes = 1;
    }

    IPluginV2DynamicExt *clone() const noexcept
    {
      return new DecodePlugin(_score_thresh, _top_n, _stride, _f_size,
                              _num_anchors, _anchors);
    }

  private:
    // đẩy data từ val vào buffer --- Trong khi save model
    template <typename T>
    void write(char *&buffer, const T &val) const
    {
      *reinterpret_cast<T *>(buffer) = val;
      buffer += sizeof(T);
    }

    // đẩy data từ buffer vào val --- Trong khi read model
    template <typename T>
    void read(const char *&buffer, T &val)
    {
      val = *reinterpret_cast<const T *>(buffer);
      buffer += sizeof(T);
    }
  };

  class DecodePluginCreator : public IPluginCreator
  {
  public:
    DecodePluginCreator() {}

    const char *getPluginName() const noexcept override
    {
      return PLUGIN_NAME;
    }

    const char *getPluginVersion() const noexcept override
    {
      return PLUGIN_VERSION;
    }

    const char *getPluginNamespace() const noexcept override
    {
      return PLUGIN_NAMESPACE;
    }

    IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength)noexcept override
    {
      return new DecodePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) noexcept override {}
    const PluginFieldCollection *getFieldNames()noexcept override { return nullptr; }
    IPluginV2DynamicExt *createPlugin(const char *name, const PluginFieldCollection *fc)noexcept override { return nullptr; }
  };

  REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);

}

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
