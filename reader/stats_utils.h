//
// Created by 孙嘉禾 on 2019/12/18.
//

#ifndef TF_OPS_STATS_UTILS_H
#define TF_OPS_STATS_UTILS_H

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace stats_utils {
extern const char kDelimiter[];
extern const char kExecutionTime[];
extern const char kThreadUtilization[];
extern const char kBufferSize[];
extern const char kBufferCapacity[];
extern const char kBufferUtilization[];
extern const char kFilteredElements[];
extern const char kDroppedElements[];
extern const char kFeaturesCount[];
extern const char kFeatureValuesCount[];
extern const char kExamplesCount[];

// Name for tf.data function execution time (in ns) histogram metrics.
string ExecutionTimeHistogramName(const string& prefix);

// Name for thread utilization (ratio of threads being used and maximum number
// of threads allocated) scalar metrics.
string ThreadUtilizationScalarName(const string& prefix);

// Name for buffer size scalar metrics.
string BufferSizeScalarName(const string& prefix);

// Name for buffer capacity (maximum allocated buffer size) scalar metrics.
string BufferCapacityScalarName(const string& prefix);

// Name for buffer utilization (ratio of buffer size and maximum allocated
// buffer size.) histogram metrics.
string BufferUtilizationHistogramName(const string& prefix);

// Name for filtered elements scalar metrics.
string FilterdElementsScalarName(const string& prefix);

// Name for dropped elements scalar mereics.
string DroppedElementsScalarName(const string& prefix);

// Name for features count histogram metrics.
string FeatureHistogramName(const string& prefix);

// Name for feature-values count histogram metrics.
string FeatureValueHistogramName(const string& prefix);


}  // namespace stats_utils
}  // namespace data
}  // namespace tensorflow



#endif //TF_OPS_STATS_UTILS_H
