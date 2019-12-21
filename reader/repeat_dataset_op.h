//
// Created by 孙嘉禾 on 2019/12/20.
//

#ifndef TF_OPS_REPEAT_DATASET_OP_H
#define TF_OPS_REPEAT_DATASET_OP_H

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

class RepeatDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char *const kDatasetType = "Repeat";
  static constexpr const char *const kInputDataset = "input_dataset";
  static constexpr const char *const kCount = "count";
  static constexpr const char *const kOutputTypes = "output_types";
  static constexpr const char *const kOutputShapes = "output_shapes";
  explicit RepeatDatasetOp(OpKernelConstruction *ctx);
 protected:
  void MakeDataset(OpKernelContext *ctx, DatasetBase *input, DatasetBase **output) override;
 private:
  class Dataset;
};

}
}

#endif //TF_OPS_REPEAT_DATASET_OP_H
