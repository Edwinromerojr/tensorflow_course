import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# create tensors
tensor_a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
tensor_b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)


# Addition
result_add = tf.add(tensor_a, tensor_b)

# Subtraction
result_sub = tf.subtract(tensor_a, tensor_b)

# Multiplication
result_mul = tf.multiply(tensor_a, tensor_b)

# Division
result_div = tf.divide(tensor_a, tensor_b)

# ----------------------------------------------

# Square
result_square = tf.square(tensor_a)

# Square root
result_sqrt = tf.sqrt(tensor_a)

# Exponential
result_exp = tf.exp(tensor_a)

# Logarithm
result_log = tf.math.log(tensor_a)

# ----------------------------------------------

# Sum along axis
result_sum = tf.reduce_sum(tensor_a, axis=1)

# Mean along axis
result_mean = tf.reduce_mean(tensor_a)

# Maximum value
result_max = tf.reduce_max(tensor_a)

# ----------------------------------------------

# Matrix multiplication
result_matmul = tf.matmul(tensor_a, tensor_b)

# Matrix transposition
result_transpose = tf.transpose(tensor_a)

# Matrix inverse (requires tensorflow addons)
result_inverse = tf.linalg.inv(tensor_a)

# ----------------------------------------------

# Accessing specific element
element = tensor_a[0, 0]

# Slicing
slice_tensor = tensor_a[:, 1]


print(result_add)
print(result_sub)
print(result_mul)
print(result_div)
print("------------------------------")
print(result_square)
print(result_sqrt)
print(result_exp)
print(result_log)
print("------------------------------")
print(result_sum)
print(result_mean)
print(result_max)
print("------------------------------")
print(result_matmul)
print(result_transpose)
print(result_inverse)
print("------------------------------")
print(element)
print(slice_tensor)


print("------------------------------")

# Broadcasting in arithmetic operations
tensor_c = tf.constant([1, 2], dtype=tf.float32)
resul_broadcast = tensor_a + tensor_c

print(resul_broadcast)