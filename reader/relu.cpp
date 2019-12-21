//
// Created by 孙嘉禾 on 2019/12/8.
//

#include "relu.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.h"

namespace tensorflow {
REGISTER_OP("Reelu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: {realnumbertype, qint8}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("ReeluGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

using FDH = FunctionDefHelper;
using CPUDevice = Eigen::ThreadPoolDevice;

#define REGISTER_RELU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Reelu").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      ReluOp<CPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                \
  Name("ReeluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
  ReluGradOp<CPUDevice, type>);

REGISTER_RELU_KERNELS(float);
REGISTER_RELU_KERNELS(double);
REGISTER_RELU_KERNELS(int32);

#undef REGISTER_RELU_KERNELS


//Status ReeluGrad(const AttrSlice& attrs, FunctionDef* g) {
//    // clang-format off
//    *g = FDH::Define(
//        // Arg defs
//        {"x: T", "dy: T"},
//        // Ret val defs
//        {"dx: T"},
//        // Attr defs
//        {{"T: {float, double}"}},
//        // Nodes
//        {
//            {{"dx"}, "ReeluGrad", {"dy", "x"}, {{"T", "$T"}}}
//        });
//    // clang-format on
//    return Status::OK();
//}
//REGISTER_OP_GRADIENT("Reelu", ReeluGrad);

}
