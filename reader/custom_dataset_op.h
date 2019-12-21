//
// Created by 孙嘉禾 on 2019/12/16.
//

#ifndef TF_OPS_CUSTOM_DATASET_OP_H
#define TF_OPS_CUSTOM_DATASET_OP_H

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"

REGISTER_OP("CustomReaderDataset")
    .Input("filenames: string")
    .Input("compression_type: string")
    .Input("buffer_size: int64")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* ctx){
        tensorflow::shape_inference::ShapeHandle unused;
        // filenames must a scalar or a vector
        TF_RETURN_IF_ERROR(ctx->WithRankAtMost(ctx->input(0), 1, &unused));
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 0, &unused));
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(2), 0, &unused));
        return tensorflow::shape_inference::ScalarShape(ctx);
    });

namespace tensorflow {
namespace data {
class CustomDatasetOp : public DatasetOpKernel {
 public:
  static constexpr const char *const kDatasetType = "Custom";
  static constexpr const char *const kFileNames = "filenames";
  static constexpr const char *const kCompressionType = "compression_type";
  static constexpr const char *const kBufferSize = "buffer_size";
  explicit CustomDatasetOp(OpKernelConstruction *ctx);
 protected:
  void MakeDataset(OpKernelContext *ctx, DatasetBase **output) override;
 private:
  class Dataset;
};

}
}

#endif //TF_OPS_CUSTOM_DATASET_OP_H
