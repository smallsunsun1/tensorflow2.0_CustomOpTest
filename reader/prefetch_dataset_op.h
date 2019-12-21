//
// Created by 孙嘉禾 on 2019/12/18.
//

#ifndef TF_OPS_PREFETCH_DATASET_OP_H
#define TF_OPS_PREFETCH_DATASET_OP_H

#include "tensorflow/core/framework/dataset.h"
#include "prefetch_autotuner.h"

namespace tensorflow {
namespace data {

class PrefetchDatasetOp: public UnaryDatasetOpKernel{
 public:
  static constexpr const char* const kDatasetType = "Prefetch";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kBufferSize = "buffer_size";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kSlackPeriod = "slack_period";
  static constexpr const char* const kLegacyAutotune = "legacy_autotune";
  explicit PrefetchDatasetOp(OpKernelConstruction* ctx);
 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input, DatasetBase** output) override ;
 private:
  class Dataset;
  int64 slack_period_ = 0;
  bool legacy_autotune_ = true;
};

}
}

#endif //TF_OPS_PREFETCH_DATASET_OP_H
