import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

def part1():
    x = [[2.]]  # No need for placeholders!
    m = tf.matmul(x, x)
    
    print(m)  # No sessions!
    # tf.Tensor([[4.]], shape=(1, 1), dtype=float32)

part1()