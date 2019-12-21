//
// Created by 孙嘉禾 on 2019/12/18.
//

#include "prefetch_autotuner.h"

#include "tensorflow/core/framework/model.h"

namespace tensorflow{
namespace data{
PrefetchAutotuner::PrefetchAutotuner(tensorflow::int64 initial_buffer_size): buffer_limit_(initial_buffer_size){
    if (initial_buffer_size == model::kAutotune){
        mode_ = Mode::kUpswing;
        buffer_limit_ = 1;
    }
}

namespace {
size_t kBufferLimitThreshold = 2048;
}

void PrefetchAutotuner::RecordConsumption(size_t current_buffer_size) {
    switch (mode_) {
        case Mode::kDisabled :return;
        case Mode::kUpswing :
            if (current_buffer_size == buffer_limit_){
                mode_ = Mode::kDownswing;
            }
            return;
        case Mode::kDownswing:
            if (current_buffer_size == 0) {
                if (buffer_limit_ >= kBufferLimitThreshold) {
                    buffer_limit_ += kBufferLimitThreshold;
                } else {
                    buffer_limit_ *= 2;
                }
                mode_ = Mode::kUpswing;
            }
            return;
    }
}

}
}
