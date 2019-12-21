//
// Created by 孙嘉禾 on 2019/12/20.
//

#include "repeat_dataset_op.h"

#include "name_utils.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"

REGISTER_OP("CustomRepeatDataset")
    .Input("input_dataset: variant")
    .Input("count: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      tensorflow::shape_inference::ShapeHandle count_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &count_shape));
      return tensorflow::shape_inference::ScalarShape(c);
    });

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char *const RepeatDatasetOp::kDatasetType;
/* static */ constexpr const char *const RepeatDatasetOp::kInputDataset;
/* static */ constexpr const char *const RepeatDatasetOp::kCount;
/* static */ constexpr const char *const RepeatDatasetOp::kOutputTypes;
/* static */ constexpr const char *const RepeatDatasetOp::kOutputShapes;

constexpr char kForeverRepeat[] = "ForeverRepeat";
constexpr char kEmptyRepeat[] = "EmptyRepeat";
constexpr char kFiniteRepeat[] = "FiniteRepeat";
constexpr char kCurIteration[] = "i";
constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kUninitialized[] = "uninitialized";
constexpr int64 kKnownRatio = 1;

class RepeatDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext *ctx, const DatasetBase *input, int64 count) : DatasetBase(DatasetContext(ctx)),
                                                                         count_(count), input_(input) {
      input_->Ref();
  }
  ~Dataset() override { input_->Unref(); }
  std::unique_ptr<IteratorBase> MakeIteratorInternal(const string &prefix) const override {
      if (count_ == 0) {
          return absl::make_unique<EmptyIterator>(EmptyIterator::Params{
              this, name_utils::IteratorPrefix(kEmptyRepeat, prefix)
          });
      } else {
          return absl::make_unique<FiniteIterator>(FiniteIterator::Params{
              this, name_utils::IteratorPrefix(kFiniteRepeat, prefix)
          });
      }
  }
  const DataTypeVector &output_dtypes() const override {
      return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape> &output_shapes() const override {
      return input_->output_shapes();
  }
  string DebugString() const override {
      return name_utils::DatasetDebugString(RepeatDatasetOp::kDatasetType);
  }
  int64 Cardinality() const override {
      int64 n = input_->Cardinality();
      if (count_ < 0) {
          if (n == 0) {
              return 0;
          }
          return kInfiniteCardinality;
      }
      if (count_ == 0) {
          return 0;
      }
      if (n == kInfiniteCardinality || n == kUnknownCardinality) {
          return n;
      }
      return count_ * n;
  }
 protected:
  Status AsGraphDefInternal(SerializationContext *ctx, DatasetGraphDefBuilder *b,
                            Node **output) const override {
      Node *input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node *count = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(count_, &count));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node, count}, output));
      return Status::OK();
  }
 private:
  class EmptyIterator : public DatasetIterator<Dataset> {
   public:
    explicit EmptyIterator(const Params &params) : DatasetIterator<Dataset>(params) {}
    Status GetNextInternal(IteratorContext *ctx, std::vector<Tensor> *out_tensors,
                           bool *end_of_sequence) override {
        *end_of_sequence = true;
        return Status::OK();
    }
   protected:
    std::shared_ptr<model::Node> CreateNode(IteratorContext *ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args), kKnownRatio);
    }
    Status SaveInternal(IteratorStateWriter *writer) override {
        return Status::OK();
    }
    Status RestoreInternal(IteratorContext *ctx, IteratorStateReader *reader) override {
        return Status::OK();
    }
  };
  class FiniteIterator : public DatasetIterator<Dataset> {
   public:
    explicit FiniteIterator(const Params &params) : DatasetIterator<Dataset>(params), i_(0) {}
    Status Initialize(IteratorContext *ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
    }
    Status GetNextInternal(IteratorContext *ctx, std::vector<Tensor> *out_tensors, bool *end_of_sequence) override {
        mutex_lock l(mu_);
        if (!input_impl_) {
            *end_of_sequence = true;
            return Status::OK();
        }
        while (i_ < dataset()->count_) {
            TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
            if (!(*end_of_sequence)) {
                return Status::OK();
            }
            ++i_;
            TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
        }
        *end_of_sequence = true;
        input_impl_.reset();
        return Status::OK();
    }
   protected:
    std::shared_ptr<model::Node> CreateNode(IteratorContext *ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args), kKnownRatio);
    }
    Status SaveInternal(IteratorStateWriter *writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurIteration), i_));
        if (!input_impl_) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kInputImplEmpty), ""));
        } else {
            TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        }
        return Status::OK();
    }
    Status RestoreInternal(IteratorContext *ctx, IteratorStateReader *reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurIteration), &i_));
        if (!reader->Contains(full_name(kInputImplEmpty))) {
            TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        } else {
            input_impl_.reset();
        }
        return Status::OK();
    }
   private:
    mutex mu_;
    int64 i_ GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
  };
  class InfinityIterator : public DatasetIterator<Dataset> {
   public:
    explicit InfinityIterator(const Params &params) : DatasetIterator<Dataset>(params) {}
    Status Initialize(IteratorContext *ctx) override {
        mutex_lock l(mu_);
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
    }
    Status GetNextInternal(IteratorContext *ctx, std::vector<Tensor> *out_tensors, bool *end_of_sequence) override {
        mutex_lock l(mu_);
        do {
            if (!input_impl_) {
                TF_RETURN_IF_ERROR(
                    dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
            }
            Status s = input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
            DCHECK(!*end_of_sequence || out_tensors->empty());
            if (first_call_ && *end_of_sequence) {
                input_impl_.reset();
                return Status::OK();
            }
            first_call_ = false;
            if (!*end_of_sequence) {
                return s;
            } else {
                input_impl_.reset();
                first_call_ = true;
            }
        } while (true);
    }
   protected:
    std::shared_ptr<model::Node> CreateNode(IteratorContext *ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args), kKnownRatio);
    }
    Status SaveInternal(IteratorStateWriter *writer) override {
        mutex_lock l(mu_);
        if (!first_call_)
            TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        else
            TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kUninitialized), ""));
        return Status::OK();
    }
    Status RestoreInternal(IteratorContext *ctx, IteratorStateReader *reader) override {
        mutex_lock l(mu_);
        if (reader->Contains(full_name(kUninitialized))) {
            input_impl_.reset();
            first_call_ = true;
        } else {
            TF_RETURN_IF_ERROR(
                dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_)
            );
            TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
            first_call_ = false;
        }
        return Status::OK();
    }
   private:
    mutex mu_;
    bool first_call_ GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
  };
  const int64 count_;
  const DatasetBase *const input_;
};

RepeatDatasetOp::RepeatDatasetOp(tensorflow::OpKernelConstruction *ctx) : UnaryDatasetOpKernel(ctx) {}
void RepeatDatasetOp::MakeDataset(tensorflow::OpKernelContext *ctx,
                                  tensorflow::DatasetBase *input,
                                  tensorflow::DatasetBase **output) {
    int64 count;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kCount, &count));
    *output = new Dataset(ctx, input, count);
}

REGISTER_KERNEL_BUILDER(Name("CustomRepeatDataset").Device(DEVICE_CPU), RepeatDatasetOp);

}
}
