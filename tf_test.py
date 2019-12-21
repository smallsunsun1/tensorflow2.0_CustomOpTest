import tensorflow as tf
import time
from tensorflow.python.framework import ops
from tensorflow.python.data.ops import dataset_ops


custom_ops = tf.load_op_library("/Users/sunjiahe/CLionProjects/tf_ops/cmake-build-release/libops.dylib")


class CustomDatasetOp(dataset_ops.DatasetSource):
    def __init__(self, filenames, compression_type=None, buffer_size=None):
        self.filenames_ = filenames
        self.compression_type = tf.convert_to_tensor(compression_type, tf.string)
        self.buffer_size = tf.convert_to_tensor(buffer_size, tf.int64)
        variant_tensor = custom_ops.custom_reader_dataset(self.filenames_, self.compression_type,
                                                          self.buffer_size)
        super(CustomDatasetOp, self).__init__(variant_tensor)

    @property
    def element_spec(self):
        return tf.TensorSpec([], tf.dtypes.string)

class CustomRepeatDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that repeats its input several times."""

    def __init__(self, input_dataset, count):
        """See `Dataset.repeat()` for details."""
        self._input_dataset = input_dataset
        if count is None:
            self._count = tf.constant(-1, dtype=tf.int64, name="count")
        else:
            self._count = tf.convert_to_tensor(
                count, dtype=tf.int64, name="count")
        variant_tensor = custom_ops.custom_repeat_dataset(
            input_dataset._variant_tensor,  # pylint: disable=protected-access
            count=self._count,
            **self._flat_structure)
        super(CustomRepeatDataset, self).__init__(input_dataset, variant_tensor)

@ops.RegisterGradient("ToZeros")
def to_zeros_grad(op, grad):
    """
    The gradients for to_zeros
    """
    to_zero = op.inputs[0]
    to_zero_grad = tf.zeros_like(to_zero)
    return [to_zero_grad]


@ops.RegisterGradient("AddOne")
def add_one_grad(op, grad):
    return [custom_ops.add_one_grad(grad)]

@ops.RegisterGradient("Reelu")
def reelu_grad(op, grad):
    input = op.inputs[0]
    return [custom_ops.reelu_grad(grad, input)]

with tf.GradientTape() as tape:
    a = tf.Variable(tf.random.uniform(shape=[300, 300], minval=-10, maxval=10), trainable=True)
    tape.watch(a)
    b = custom_ops.to_zeros(a)
    c = tf.square(a)
    d = custom_ops.add_one(a)
    start = time.time()
    # for i in range(100):
    e = custom_ops.reelu(a)
    # f = tf.nn.relu(a)
    end = time.time()
    print(end - start)
grad1 = tape.gradient(e, a)
# print(a)
# print(e)
print(grad1)

# dataset = CustomDatasetOp("/Users/sunjiahe/PycharmProjects/CycleGan/train.sh", "", 256*1024)
# dataset = CustomDatasetOp("../train.sh", "", 256*1024)
# dataset = CustomRepeatDataset(dataset, 2)
# for ele in dataset:
#     print(ele)
