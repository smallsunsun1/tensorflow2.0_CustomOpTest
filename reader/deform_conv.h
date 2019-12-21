//
// Created by 孙嘉禾 on 2019/12/17.
//

#ifndef TF_OPS_DEFORM_CONV_H
#define TF_OPS_DEFORM_CONV_H

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#ifdef GOOGLE_CUDA
#include "tensorflow/core/util/gpu_kernel_helper.h"
#endif

namespace tensorflow {

template<typename device, typename T>
class DeformableConvolutionOp;

template <typename device, typename T>
struct DeformableIm2Col{
  void operator()(const device& d, typename TTypes<T, 4>::ConstTensor data_im,
      typename TTypes<T, 4>::ConstTensor data_offset, const int channels,
      const int height, const int width, const int ksize_h, const int ksize_w,
      const int pad_h, const int pad_w, const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w, const int parallel_imgs,
      const int deformable_group, typename TTypes<T, 4>::ConstTensor data_col);
};

#ifdef GOOGLE_CUDA
template <typename Eigen::GpuDevice, typename T>
struct DeformableIm2Col{
  void operator()(const Eigen::GpuDevice& d);
};
#endif


}

#endif //TF_OPS_DEFORM_CONV_H
