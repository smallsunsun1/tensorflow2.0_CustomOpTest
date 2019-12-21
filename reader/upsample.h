//
// Created by 孙嘉禾 on 2019-08-29.
//

#ifndef TF_OPS_UPSAMPLE_H
#define TF_OPS_UPSAMPLE_H

#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"


namespace tensorflow {
inline float CalculateResizeScale(int64 in_size, int64 out_size, bool align_corners) {
    return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                           : in_size / static_cast<float>(out_size);
}

struct LegacyScaler {
  LegacyScaler() = default;
  inline float operator()(const int x, const float scala) const {
      return (static_cast<float>(x) + 0.5f) * scala - 0.5f;
  }
};

template<bool value>
struct BoolToScaler{};

struct HalfPixelScaler {
  HalfPixelScaler() = default;
  inline float operator()(const int x, const float scale) const {
      return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
  }
};

struct ImageResizerState {
  explicit ImageResizerState(bool align_corners, bool half_pixel_centers) : align_corners_(align_corners),
                                                                            half_pixel_centers_(half_pixel_centers) {}
  /*
   * ValidateAndCalculateOutputSize checks the bounds on the input tensors
   * and requested size, sets up some of the resizing the state such as the
   * height_scale and width_scale, and calculates the output size.
   * If ant of these operations fails, it sets an error status in
   * the context, which the caller must check.
   */
  void ValidateAndCalculateOutputSize(OpKernelContext *context,
                                      const Tensor &input) {
      OP_REQUIRES(context, !half_pixel_centers_ || (half_pixel_centers_ && !align_corners_),
                  errors::InvalidArgument("If half_pixel centers is True, "
                                          "align_corners must be False"));
      const Tensor &shape_t = context->input(1);
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          shape_t.shape().DebugString()));
      OP_REQUIRES(context, shape_t.NumElements() == 2,
                  errors::InvalidArgument("shape_t must have two elements",
                                          shape_t.shape().DebugString()));
      auto Svec = shape_t.vec<int32>();
      batch_size = input.dim_size(0);
      out_height = Svec(0);
      out_width = Svec(1);
      in_height = static_cast<int32>(input.dim_size(1));
      in_width = static_cast<int32>(input.dim_size(2));
      channels = input.dim_size(3);
      OP_REQUIRES(context, out_height > 0 && out_width > 0,
                  errors::InvalidArgument("output dimensions must be positive"));
      OP_REQUIRES(context, channels > 0,
                  errors::InvalidArgument("image must have at least one channel"));
      OP_REQUIRES(context, input.dim_size(1) > 0 && input.dim_size(2) > 0,
                  errors::InvalidArgument("input image must be of non-zero size"));
      height_scale = CalculateResizeScale(in_height, out_height, align_corners_);
      width_scale = CalculateResizeScale(in_width, out_width, align_corners_);

      // Guard against overflows
      OP_REQUIRES(context, ceilf((out_height - 1) * height_scale)
          <= static_cast<float>(std::numeric_limits<int64>::max()),
                  errors::InvalidArgument("input image height scale would cause an overflow"));
      OP_REQUIRES(context,
                  ceilf((out_width - 1) * width_scale) <= static_cast<float>(INT_MAX),
                  errors::InvalidArgument(
                      "input image width scale would cause an overflow"));
  }
  void ValidateAndCreateOutput(OpKernelContext *context, const Tensor &input) {
      ValidateAndCalculateOutputSize(context, input);
      if (!context->status().ok())
          return;
      OP_REQUIRES_OK(context, context->allocate_output(0,
                                                       TensorShape({input.dim_size(0), out_height, out_width,
                                                                    input.dim_size(3)}),
                                                       &output));
  }

  int64 batch_size;
  int64 out_height;
  int64 out_width;
  int64 in_height;
  int64 in_width;
  int64 channels;
  float height_scale;
  float width_scale;
  Tensor *output = nullptr;
 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

struct ImageResizerGradientState {
  explicit ImageResizerGradientState(bool align_corners, bool half_pixel_centers) :
      align_corners_(align_corners), half_pixel_centers_(half_pixel_centers) {}
  void ValidateAndCreateOutput(OpKernelContext *context, const Tensor &input, const Tensor &original_image) {
      OP_REQUIRES(context, !half_pixel_centers_ || (half_pixel_centers_ && !align_corners_),
                  errors::InvalidArgument("If half_pixel_centers is True, "
                                          "align_corners must be false."));
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input_grad must be 4-dimensional",
                                          input.shape().DebugString()));
      // Resizers always produce float images, so input gradient must always be a float.
      OP_REQUIRES(context, input.dtype() == DT_FLOAT,
                  errors::InvalidArgument("input_grad must be of type float",
                                          DataTypeString(input.dtype())));
      OP_REQUIRES(context, original_image.dims() == 4,
                  errors::InvalidArgument("original_image must be 4-dimensional",
                                          original_image.shape().DebugString()));
      batch_size = input.dim_size(0);
      channels = input.dim_size(3);
      resized_height = input.dim_size(1);
      resized_width = input.dim_size(2);
      original_height = original_image.dim_size(1);
      original_width = original_image.dim_size(2);
      height_scale = CalculateResizeScale(original_height, resized_height, align_corners_);
      width_scale = CalculateResizeScale(original_width, resized_width, align_corners_);
      output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0,
                                                       TensorShape({batch_size, original_height,
                                                                    original_width, channels}), &output));
  }
  int64 batch_size;
  int64 channels;
  int64 resized_height;
  int64 resized_width;
  int64 original_height;
  int64 original_width;
  float height_scale;
  float width_scale;
  Tensor *output;
 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

namespace functor {
template<typename Device, typename T, bool half_pixel_center, bool align_corners>
struct ResizeNearestNeighbor {
  bool operator()(const Device &d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output);
};

template<typename Device, typename T, bool half_pixel_centers, bool align_corners>
struct ResizeNearestNeighborGrad {
  bool operator()(const Device &d, typename TTypes<T, 4>::ConstTensor input_grad,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output_grad);
};

}

}

#endif //TF_OPS_UPSAMPLE_H
