import os
import tensorflow.compat.v1 as tf


# restore_path = str(os.getenv('RESTORE_PATH', '/home/position_0.ckpt'))
saver_path = str(os.getenv('SAVER_PATH', '/home/postion/'))

# Define the input placeholders
input_numerical = tf.placeholder(tf.float32, shape=[None, 40], name='input_numerical')


# Define the neural network layers
hidden_layer_size = 256
output_size = 100
hidden_layer = tf.layers.dense(input_numerical, hidden_layer_size, activation=tf.nn.relu, name='hidden_layer')
output_layer = tf.layers.dense(hidden_layer, output_size, activation=None, name='output_layer')

# Define the loss function
output_true = tf.placeholder(tf.int32, shape=[None], name='output_true')
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_layer, labels=output_true, name='cross_entropy')
loss = tf.reduce_mean(cross_entropy, name='loss')

# Define the optimizer and training operation
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss, name='train_op')

# Define the accuracy metric
predicted_labels = tf.argmax(output_layer, axis=1, output_type=tf.int32, name='predicted_labels')
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, output_true), tf.float32), name='accuracy')
