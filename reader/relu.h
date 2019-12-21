//
// Created by 孙嘉禾 on 2019/12/8.
//

#ifndef TF_OPS_RELU_H
#define TF_OPS_RELU_H

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

template<typename T>
struct ReluOpCPUImpl {
  void operator()(OpKernelContext *ctx, typename TTypes<T>::ConstFlat input_flat,
                  typename TTypes<T>::Flat output_flat, int num_split = 4) {
      auto f = std::function<void(int64, int64)>([&input_flat, &output_flat](int64 start, int64 end) {
        for (auto i = start; i < end; i++) {
            if (input_flat(i) > 0) {
                output_flat(i) = input_flat(i);
            } else
                output_flat(i) = 0;
        }
      });
      ctx->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(num_split, input_flat.size() / num_split, f);
//      f(0, input_flat.size());
//      pool->ParallelFor(num_split,
//                        input_flat.size() / num_split,
//                        f);
  }
};

template<typename Device, typename T>
class ReluOp : public OpKernel {
 public:
  explicit ReluOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext *ctx) override {
      const Tensor &input = ctx->input(0);
      Tensor *output = nullptr;
      TensorShape input_shape = input.shape();
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_shape, &output));
      auto input_flat = input.flat<T>();
      auto output_flat = output->flat<T>();
//      for (int i = 0; i < input_flat.size(); i++){
//          if (input_flat(i) > 0)
//              output_flat(i) = input_flat(i);
//          else
//              output_flat(i) = 0;
//      }
      ReluOpCPUImpl<T>{}(ctx, input_flat, output_flat);
//      pool->ParallelFor(input_flat.size(),
//                        thread::ThreadPool::SchedulingParams{thread::ThreadPool::SchedulingStrategy::kAdaptive, {}, {}},
//                        f);
  }
};

template<typename Device, typename T>
class ReluGradOp : public OpKernel {
 public:
  explicit ReluGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext *ctx) override {
      const Tensor &output_backprop = ctx->input(0);
      const Tensor &input = ctx->input(1);
      OP_REQUIRES(ctx, output_backprop.NumElements() < std::numeric_limits<int32>::max(),
                  errors::InvalidArgument("Grad requires tensor size <= int32 max"));
      Tensor *output = nullptr;
      TensorShape output_shape = output_backprop.shape();
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
      auto output_flat = output->flat<T>();
      auto output_backprop_flat = output_backprop.flat<T>();
      auto input_flat = input.flat<T>();
      ctx->device()->tensorflow_cpu_worker_threads()->workers->TransformRangeConcurrently(4,
                                       input_flat.size(),
                                       [&input_flat, &output_flat, &output_backprop_flat](int64 start, int64 end) {
                                         for (int i = start; i < end; i++) {
                                             if (input_flat(i) > 0)
                                                 output_flat(i) = output_backprop_flat(i);
                                             else
                                                 output_flat(i) = 0;
                                         }
                                       });
//      for (int i = 0; i < input_flat.size(); i++){
//          if (input_flat(i) > 0)
//              output_flat(i) = output_backprop_flat(i);
//          else
//              output_flat(i) = 0;
//      }
  }
};

}

#endif //TF_OPS_RELU_H
