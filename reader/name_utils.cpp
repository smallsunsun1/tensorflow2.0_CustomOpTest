//
// Created by 孙嘉禾 on 2019/12/18.
//

#include "name_utils.h"

#include <absl/strings/str_join.h>

namespace tensorflow {
namespace data {
namespace name_utils {

ABSL_CONST_INIT const char kDelimiter[] = "::";
ABSL_CONST_INIT const char kDefaultDatasetDebugStringPrefix[] = "";

constexpr char kDataset[] = "Dataset";
constexpr char kOp[] = "Op";
constexpr char kVersion[] = "V";

string OpName(const string& dataset_type) {
    return OpName(dataset_type, OpNameParams());
}

string OpName(const string& dataset_type, const OpNameParams& params) {
    if (params.op_version == 1) {
        return strings::StrCat(dataset_type, kDataset);
    }
    return strings::StrCat(dataset_type, kDataset, kVersion, params.op_version);
}

string ArgsToString(const std::vector<string>& args) {
    if (args.empty()) {
        return "";
    }
    return strings::StrCat("(", absl::StrJoin(args, ", "), ")");
}

string DatasetDebugString(const string& dataset_type) {
    return DatasetDebugString(dataset_type, DatasetDebugStringParams());
}

string DatasetDebugString(const string& dataset_type,
                          const DatasetDebugStringParams& params) {
    OpNameParams op_name_params;
    op_name_params.op_version = params.op_version;
    string op_name = OpName(dataset_type, op_name_params);
    return strings::StrCat(op_name, kOp, ArgsToString(params.args), kDelimiter,
                           params.dataset_prefix, kDataset);
}

string IteratorPrefix(const string& dataset_type, const string& prefix) {
    return IteratorPrefix(dataset_type, prefix, IteratorPrefixParams());
}

string IteratorPrefix(const string& dataset_type, const string& prefix,
                      const IteratorPrefixParams& params) {
    if (params.op_version == 1) {
        return strings::StrCat(prefix, kDelimiter, params.dataset_prefix,
                               dataset_type);
    }
    return strings::StrCat(prefix, kDelimiter, params.dataset_prefix,
                           dataset_type, kVersion, params.op_version);
}

}  // namespace name_utils
}  // namespace data
}  // namespace tensorflow

