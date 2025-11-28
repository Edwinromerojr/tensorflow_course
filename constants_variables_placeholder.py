import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Constants / Can't be changed

# Create a constant tensor
constant_tensor = tf.constant([1, 2, 3, 4, 5])

# Variables / Can be changed

# Create a variable tensor with initial values
initial_values = tf.random.normal(shape=(3, 3))
variable_tensor = tf.Variable(initial_values)

# Update variable tensor
new_values = tf.random.normal(shape=(3, 3))
variable_tensor.assign(new_values) 

print("------------------------------")
print("Constant Tensor:", constant_tensor)
print("------------------------------")
print("Variable Tensor after update:", variable_tensor)
print("------------------------------")
print("------------------------------")

# Placeholders (Deprecated)

# # Create a placeholder for input data (Deprecated)
# x = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))

# # Placeholder usage in computation graph
# y = tf.square(x)

# TensorFlow 2.x approach
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Direct computation without placeholders
y = tf.square(x)

print(x)
print(y)