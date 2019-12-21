//
// Created by 孙嘉禾 on 2019/12/17.
//

#include "deform_conv.h"

REGISTER_OP("DeformableConvolution")
    .Input("input: T")
    .Input("weight: T")
    .Input("offset: T")
    .Output("output: T")
    .Attr("T: {float, double, half}")
    .Attr("kw: int")
    .Attr("kh: int")
    .Attr("dw: int")
    .Attr("dh: int")
    .Attr("padw: int")
    .Attr("padh: int")
    .Attr("dilation_w: int")
    .Attr("dilation_h: int")
    .Attr("group: int")
    .Attr("deformable_group: int")
    .Attr("im2col_step: int")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *ctx) {
      tensorflow::shape_inference::ShapeHandle handle;
      auto input_shape_handle = ctx->input(0);
      ctx->set_output(0, input_shape_handle);
      return tensorflow::Status::OK();
    });

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// Deformable Convolution Implementation, assume input is [B, C, H, W]
template<typename device, typename T>
class DeformableConvolutionOp : public OpKernel {
  explicit DeformableConvolutionOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("kw", &kw_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("kh", &kh_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("dw", &dw_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("dw", &dw_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("dh", &dh_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("padw", &padw_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("padh", &padh_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("dilation_w", &dilation_w));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("dilation_h", &dilation_h));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("group", &group_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("deformable_group", &deformable_group_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("im2col_step", &im2col_step_));
  }
  void Compute(OpKernelContext *ctx) override {
      const auto &input = ctx->input(0);
      const auto &weight = ctx->input(1);
      const auto &offset = ctx->input(2);
      int batch = 1;
      auto batchSize = input.dim_size(0);
      auto nInputPlane = input.dim_size(1);
      auto inputHeight = input.dim_size(2);
      auto inputWidth = input.dim_size(3);
      auto nOutputPlane = weight.dim_size(0);
      int64 outputWidth = (inputWidth * 2 * padw_ - (dilation_w * (kw_ - 1) + 1)) / dw_ + 1;
      int64 outputHeight = (inputHeight + 2 * padh_ - (dilation_h * (kh_ - 1) + 1)) / dh_ + 1;
      OP_REQUIRES(ctx, offset.dim_size(0) == batchSize, errors::InvalidArgument("invalid batch size of offset"));
      Tensor *output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batchSize / im2col_step_, im2col_step_, nOutputPlane, outputHeight,
                               outputWidth}, &output));
      Tensor columns;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataType::DT_FLOAT,
                         {nInputPlane * kw_ * kh_, im2col_step_ * outputHeight * outputWidth},
                         &columns));
      auto columns_flat = columns.tensor<T, 4>();
      columns_flat.setZero();
      Tensor ones;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataType::DT_FLOAT, {outputHeight, outputWidth}, &ones));
      auto input_flat = input.shaped<T, 5>({batchSize / im2col_step_, im2col_step_, nInputPlane,
                                            inputHeight, inputWidth});
      auto offset_flat = offset.shaped<T, 5>({batchSize / im2col_step_, im2col_step_, deformable_group_ * 2 * kh_ * kw_,
                                              outputHeight, outputWidth});
      Tensor output_buffer;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataType::DT_FLOAT, {batchSize / im2col_step_, nOutputPlane,
                                              im2col_step_ * outputHeight, outputWidth}, &output_buffer));
      auto output_buffer_flat = output_buffer.shaped<T, 5>(output_buffer.dim_size(0), group_, output_buffer.dim_size(1) / group_,
                                                 output_buffer.dim_size(2), output_buffer.dim_size(3));
      for (int elt = 0; elt < batchSize / im2col_step_; elt++) {

      }
  }
 private:
  int kw_;
  int kh_;
  int dw_;
  int dh_;
  int padw_;
  int padh_;
  int dilation_w;
  int dilation_h;
  int group_;
  int deformable_group_;
  int im2col_step_;
};


template <typename T>
struct DeformableIm2Col<CPUDevice, T>{
  void operator()(const CPUDevice & d, typename TTypes<T, 4>::ConstTensor data_im,
  typename TTypes<T, 4>::ConstTensor data_offset, const int channels,
  const int height, const int width, const int ksize_h, const int ksize_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w, const int parallel_imgs,
  const int deformable_group, typename TTypes<T, 4>::ConstTensor data_col){
      // num_axes should be smaller than block size
      // TODO: check parallel_imgs is correctly passed in
      int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
      int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
      int num_kernels = channels * height_col * width_col * parallel_imgs;
      int channel_per_deformable_group = channels / deformable_group;

  }
};


}



