#include <iostream>
#include <fstream>
#include <memory>
#include "tensorflow/core/framework/resource_mgr.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/dataset.h"

//template<typename T>
//class Animal {
// public:
//  void talk() {
//      static_cast<T*>(this)->talkImplement();
//  }
//};
//class Cat : public Animal<Cat> {
// public:
//  void talkImplement() {
//      std::cout << "Miao \n";
//  }
//};
//class Developer : public Animal<Developer> {
// public:
//  void talkImplement() {
//      std::cout << "Hello World \n";
//  }
//};
//template<typename T>
//void LetAnimalTalk(Animal<T> &pa) {
//    pa.talk();
//}

struct in_place_t {};
static constexpr in_place_t InPlace() {
    return in_place_t{};
}

struct ValueInterface {
  virtual ~ValueInterface() = default;
  virtual void CloneInto(ValueInterface *memory) const = 0;
};

template<typename T>
struct Value final : ValueInterface {
  template<typename ...Args>
  explicit Value(in_place_t, Args &&... args):value(std::forward<Args>(args)...) {}
  ~Value() = default;
  void CloneInto(ValueInterface *memory) const final {
      new(memory) Value(InPlace(), value);
  }
  T value;
};

namespace tensorflow {
struct MyVar : public ResourceBase {
  string DebugString() const {
      return "Error in MyVar";
  }
  mutex mu;
  Tensor val;
};
}

int main() {
    tensorflow::Tensor t(tensorflow::DataType::DT_FLOAT, {2, 2});
    auto flat = t.flat<float>();
    flat.setZero();
    auto func = [&flat](int i) {flat(i) = 300;};
    auto& flat_ref = flat;
    flat_ref(2) = 100;
    func(3);
    std::cout << flat_ref << std::endl;
    std::cout << flat << std::endl;
//    Eigen::Tensor<int, 2> t(200, 200);
//    t.setConstant(2);
//    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < 1000; i++) {
//        Eigen::Tensor<int, 2> temp = t * t;
//    }
//    auto end = std::chrono::high_resolution_clock::now();
//    auto duration = end - start;
//    std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0) << std::endl;
//    tensorflow::ResourceMgr rm;
//    tensorflow::MyVar *my_var = new tensorflow::MyVar;
//    my_var->val = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 2, 3});
//    my_var->val.flat<float>().setZero();  // 0 initialized
//    auto a = std::make_unique<int[]>(5);


//    std::string out = "/Users/sunjiahe/CLionProjects/tf_ops/data";
//    std::fstream output(out, std::ios::out | std::ios::binary);
//    SearchRequest request;
//    request.set_result_per_page(1);
//    request.set_page_number(12);
//    request.set_query("sss");
//    std::string values;
//    request.SerializeToString(&values);
//    request.SerializePartialToOstream(&output);
//    output.close();
//    SearchRequest another;
//    another.ParseFromString(values);
//    std::cout << another.query() << std::endl;
//    std::cout << another.page_number() << std::endl;
//    std::cout << another.result_per_page() << std::endl;
    return 0;
}