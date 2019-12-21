//
// Created by 孙嘉禾 on 2019-08-29.
//

#include "upsample.h"
#include <memory>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

Status SetOutputToSizedImage(shape_inference::InferenceContext *c, shape_inference::DimensionHandle batch_dim,
                             int size_input_idx, shape_inference::DimensionHandle channel_dim) {
    // Verify shape of size input
    ShapeHandle size;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(size_input_idx), 1, &size));
    DimensionHandle unused;
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 2, &unused));

    // Get size values from the size tensor
    const Tensor *size_tensor = c->input_tensor(size_input_idx);
    DimensionHandle width;
    DimensionHandle height;
    if (size_tensor == nullptr) {
        width = c->UnknownDim();
        height = c->UnknownDim();
    } else {
        // TODO - Remove once we have constant evaluation in C++ only.
        if (size_tensor->dtype() != DT_INT32) {
            return errors::InvalidArgument(
                "Bad size input type for SetOutputToSizedImage: Expected DT_INT32 "
                "but got ",
                DataTypeString(size_tensor->dtype()), " for input #", size_input_idx,
                " in ", c->DebugString());
        }
        auto vec = size_tensor->vec<int32>();
        height = c->MakeDim(vec(0));
        width = c->MakeDim(vec(1));
    }
    c->set_output(0, c->MakeShape({batch_dim, height, width, channel_dim}));
    return Status::OK();
}

Status ResizeShapeFn(shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
    return SetOutputToSizedImage(c, c->Dim(input, 0), 1 /* size_input_idx */,
                                 c->Dim(input, 3));
}

REGISTER_OP("ResizeNearestNeighbor")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: T")
    .Attr("T: {int8, uint8, int16, uint16, int32, int64, half, float, double}")
    .Attr("align_corners: bool = false")
    .Attr("half_pixel_centers: bool = false")
    .SetShapeFn(ResizeShapeFn);

REGISTER_OP("ResizeNearestNeighborGrad")
    .Input("grads: T")
    .Input("size: int32")
    .Output("output: T")
    .Attr("T: {uint8, int8, int32, half, float, double}")
    .Attr("align_corners: bool = false")
    .Attr("half_pixel_centers: bool = false")
    .SetShapeFn([](InferenceContext* c){
      return Status::OK();
    });

template<typename device, typename T>
class ResizeNearestNeighborOp : public OpKernel {
 public:
  explicit ResizeNearestNeighborOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("align_corners", &align_corners_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }
  void Compute(OpKernelContext *ctx) override {

      const Tensor &input = ctx->input(0);
      ImageResizerState st(align_corners_, half_pixel_centers_);
      st.ValidateAndCreateOutput(ctx, input);
      if (!ctx->status().ok())
          return;
      if (st.output->NumElements() == 0)
          return;
      typename TTypes<T, 4>::ConstTensor input_data(input.tensor<T, 4>());
      typename TTypes<T, 4>::Tensor output_data(st.output->tensor<T, 4>());
      bool status;
      if (half_pixel_centers_) {
          if (align_corners_) {
              status = functor::ResizeNearestNeighbor<device, T, true, true>()(
                  ctx->eigen_device<device>(), input_data, st.height_scale, st.width_scale,
                  output_data);
          } else {
              status = functor::ResizeNearestNeighbor<device, T, true, false>()(
                  ctx->eigen_device<device>(), input_data, st.height_scale,
                  st.width_scale, output_data);
          }
      } else {
          if (align_corners_) {
              status = functor::ResizeNearestNeighbor<device, T,
                  /*half_pixe_centers=*/false,
                  /*align_corners=*/true>()(
                  ctx->eigen_device<device>(), input_data, st.height_scale,
                  st.width_scale, output_data);
          } else {
              status = functor::ResizeNearestNeighbor<device, T,
                  /*half_pixe_centers=*/false,
                  /*align_corners=*/false>()(
                  ctx->eigen_device<device>(), input_data, st.height_scale,
                  st.width_scale, output_data);
          }
      }
      if (!status) {
          ctx->SetStatus(errors::Internal("Failed launching ResizeNearestNeighbor"));
      }
  }
 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

template<>
struct BoolToScaler<false> {
  typedef LegacyScaler Scaler;
};

template<>
struct BoolToScaler<true> {
  typedef HalfPixelScaler Scaler;
};

template<typename Device, typename T>
class ResizeNearestNeighborOpGrad : public OpKernel {
 public:
  explicit ResizeNearestNeighborOpGrad(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
      OP_REQUIRES_OK(context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }
  void Compute(OpKernelContext *context) override {

      // Grab and validate the input:
      const Tensor &input = context->input(0);
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input.shape().DebugString()));
      // Grad and validate the output shape:
      const Tensor &shape_t = context->input(1);
      OP_REQUIRES(context, shape_t.dims() == 1,
                  errors::InvalidArgument("shape_t must be 1-dimensional",
                                          shape_t.shape().DebugString()));
      auto sizes = shape_t.vec<int32>();
      OP_REQUIRES(context, sizes(0) > 0 && sizes(1) > 0,
                  errors::InvalidArgument("shape_t's elements must be positive"));
      const int64 batch_size = input.dim_size(0);
      const int64 in_height = input.dim_size(1);
      const int64 in_width = input.dim_size(2);
      const int64 channels = input.dim_size(3);
      const int64 out_height = sizes(0);
      const int64 out_width = sizes(1);
      Tensor *output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch_size, out_height, out_width, channels}),
                                                       &output));

      // Return if the output is empty
      if (output->NumElements() == 0)
          return;
      typename TTypes<T, 4>::ConstTensor input_data(input.tensor<T, 4>());
      typename TTypes<T, 4>::Tensor output_data(output->tensor<T, 4>());

      const float height_scale =
          CalculateResizeScale(out_height, in_height, align_corners_);
      const float width_scale =
          CalculateResizeScale(out_width, in_width, align_corners_);

      bool status;
      if (half_pixel_centers_) {
          if (align_corners_) {
              status = functor::ResizeNearestNeighborGrad<Device, T,
                  /*half_pixel_centers=*/true,
                  /*align_corners=*/true>()(
                  context->eigen_device<Device>(), input_data, height_scale,
                  width_scale, output_data);
          } else {
              status = functor::ResizeNearestNeighborGrad<Device, T,
                  /*half_pixel_centers=*/true,
                  /*align_corners=*/false>()(
                  context->eigen_device<Device>(), input_data, height_scale,
                  width_scale, output_data);
          }
      } else {
          if (align_corners_) {
              status =
                  functor::ResizeNearestNeighborGrad<Device, T,
                      /*half_pixel_centers=*/false,
                      /*align_corners=*/true>()(
                      context->eigen_device<Device>(), input_data, height_scale,
                      width_scale, output_data);
          } else {
              status =
                  functor::ResizeNearestNeighborGrad<Device, T,
                      /*half_pixel_centers=*/false,
                      /*align_corners=*/false>()(
                      context->eigen_device<Device>(), input_data, height_scale,
                      width_scale, output_data);
          }
      }
      if (!status) {
          context->SetStatus(
              errors::Internal("Failed launching ResizeNearestNeighborGrad"));
      }
  }
 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

namespace functor {
template<typename T, bool half_pixel_centers, bool align_corners>
struct ResizeNearestNeighbor<CPUDevice, T, half_pixel_centers, align_corners> {
  bool operator()(const CPUDevice &d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
      typename BoolToScaler<half_pixel_centers>::Scaler scaler;
      const Eigen::Index batch_size = input.dimension(0);
      const Eigen::Index in_height = input.dimension(1);
      const Eigen::Index in_width = input.dimension(2);
      const Eigen::Index channels = input.dimension(3);
      const Eigen::Index out_height = output.dimension(1);
      const Eigen::Index out_width = output.dimension(2);
      for (Eigen::Index b = 0; b < batch_size; ++b) {
          for (Eigen::Index y = 0; y < out_height; ++y) {
              Eigen::Index in_y = std::min((align_corners) ? static_cast<Eigen::Index>(roundf(scaler(y, height_scale)))
                                                           : static_cast<Eigen::Index>(floorf(scaler(y, height_scale))),
                                           in_height - 1);
              if (half_pixel_centers) {
                  in_y = std::max(static_cast<Eigen::Index>(0), in_y);
              }
              for (Eigen::Index x = 0; x < out_width; ++x) {
                  Eigen::Index in_x = std::min(
                      (align_corners)
                      ? static_cast<Eigen::Index>(roundf(scaler(x, width_scale)))
                      : static_cast<Eigen::Index>(floorf(scaler(x, width_scale))),
                      in_width - 1);
                  if (half_pixel_centers) {
                      in_x = std::max(static_cast<Eigen::Index>(0), in_x);
                  }
                  std::copy_n(&input(b, in_y, in_x, 0), channels, &output(b, y, x, 0));
              }
          }
      }
      return true;
  }
};

template<typename T, bool half_pixel_centers, bool align_corners>
struct ResizeNearestNeighborGrad<CPUDevice, T, half_pixel_centers,
                                 align_corners> {
  bool operator()(const CPUDevice &d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale, typename TTypes<T, 4>::Tensor output) {
      typename BoolToScaler<half_pixel_centers>::Scaler scaler;
      const Eigen::Index batch_size = input.dimension(0);
      const Eigen::Index in_height = input.dimension(1);
      const Eigen::Index in_width = input.dimension(2);
      const Eigen::Index channels = input.dimension(3);
      const Eigen::Index out_height = output.dimension(1);
      const Eigen::Index out_width = output.dimension(2);
      for (Eigen::Index y = 0; y < in_height; ++y) {
          const Eigen::Index out_y = std::min(
              (align_corners) ? static_cast<Eigen::Index>(roundf(scaler(y, height_scale)))
                              : static_cast<Eigen::Index>(floorf(scaler(y, height_scale))),
              out_height - 1);
          for (Eigen::Index x = 0; x < in_width; ++x) {
              const Eigen::Index out_x = std::min(
                  (align_corners)
                  ? static_cast<Eigen::Index>(roundf(scaler(x, width_scale)))
                  : static_cast<Eigen::Index>(floorf(scaler(x, width_scale))),
                  out_width - 1);
              for (Eigen::Index b = 0; b < batch_size; ++b) {
                  for (Eigen::Index c = 0; c < channels; ++c) {
                      output(b, out_y, out_x, c) += input(b, y, x, c);
                  }
              }
          }
      }
      return true;
  }
};

}

}
