//
// Created by 孙嘉禾 on 2019-08-16.
//

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("ImageThreshold")
    .Attr("T: {float, int32, uint8, double, int64}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

using CPUDevice = Eigen::ThreadPoolDevice;

template<typename Device, typename T>
class ImageThreshold : public OpKernel {
 public:
  explicit ImageThreshold(OpKernelConstruction *ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext *ctx) override {
      const Tensor &images_t = ctx->input(0);
      const Tensor &thresholds_t = ctx->input(1);
      OP_REQUIRES(ctx, thresholds_t.shape().dims() == 1,
                  errors::InvalidArgument("Thresholds must have rank 1"));
      OP_REQUIRES(ctx, thresholds_t.shape().dim_size(0) <= 255,
                  errors::InvalidArgument("Thresholds must have dim <= 255"));
      Tensor *output_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, images_t.shape(), &output_t));
      auto input_flat = images_t.flat<T>();
      auto thresholds_flat = thresholds_t.flat<T>();
      auto output_flat = output_t->flat<uint8>();

      // Set all but the first element of the output tensor to 0.
      for (int i = 0; i < input_flat.size(); i++) {
          for (int j = 0; j < thresholds_flat.size(); j++) {
              if (input_flat(i) <= thresholds_flat(j)) {
                  output_flat(i) = j;
                  break;
              }
              output_flat(i) = j + 1;
          }
      }
  }
};


#define REGISTER(TYPE)                                          \
    REGISTER_KERNEL_BUILDER(Name("ImageThreshold")              \
                                .Device(DEVICE_CPU)             \
                                .TypeConstraint<TYPE>("T"), \
                            ImageThreshold<CPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

}