//
// Created by 孙嘉禾 on 2019/12/18.
//

#ifndef TF_OPS_PREFETCH_AUTOTUNER_H
#define TF_OPS_PREFETCH_AUTOTUNER_H

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
class PrefetchAutotuner {
 public:
  explicit PrefetchAutotuner(int64 initial_buffer_size);
  int64 buffer_limit() const { return buffer_limit_; }
  void RecordConsumption(size_t current_buffer_size);
  void RecordEmpty() { RecordConsumption(0); }
 private:
  enum class Mode {
    kDisabled,
    kUpswing,
    kDownswing
  };
  int64 buffer_limit_;
  Mode mode_ = Mode::kDisabled;
};

}
}

#endif //TF_OPS_PREFETCH_AUTOTUNER_H
