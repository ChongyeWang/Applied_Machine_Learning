from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  

  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  conv4_flat = tf.reshape(conv4, [-1, 28 * 28 * 32])


  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=conv4_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  accuracy = tf.metrics.accuracy(labels=labels,
                                 predictions=predictions["classes"],
                                 name='acc_op')
  metrics = {'accuracy': accuracy}


  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    tf.summary.scalar('accuracy_train', accuracy[1])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

  # Add evaluation metrics (for EVAL mode)
  if mode == tf.estimator.ModeKeys.EVAL:
    tf.summary.scalar('accuracy_val', accuracy[1])
    return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=metrics)


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)


  # training
  training = train_data[0: int(len(train_data) * 0.8)]
  training_labels = train_labels[0: int(len(train_labels) * 0.8)]

  #validation
  validate = train_data[int(len(train_data) * 0.8):]
  validate_labels = train_labels[int(len(train_labels) * 0.8):]

  #test
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="validate")

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": training},
      y=training_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True, queue_capacity=400, num_threads=2)

  # validation and print results
  validate_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": validate},
      y=validate_labels,
      num_epochs=1,
      shuffle=False)


  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)

  for epoch in range(21):
      mnist_classifier.train(
          input_fn=train_input_fn,
          steps=100,
          # hooks=[logging_hook]
      )

      validate_results = mnist_classifier.evaluate(input_fn=validate_input_fn)
      #eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
      


if __name__ == "__main__":
    tf.app.run()