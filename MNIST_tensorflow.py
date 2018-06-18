# Tutorial from: http://adventuresinmachinelearning.com/python-tensorflow-tutorial/
# Referenced their github: https://github.com/adventuresinML/adventures-in-ml-code/blob/master/tensor_flow_tutorial.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################
### ESTABLISHING THE MODEL ###
##############################

# Setting up how the model will be trained
graphicSize = 28
numOutputs = 10
learning_rate = 0.5
epochs = 10
batch_size = 100

numUnits_hidden1 = 300

# Declare training data placeholders
#- Define input
#-- The first argument in the placeholder is the datatype of the input. We want a float, usually.
#-- The second argument in the placeholder is the dimensionality of the input.
input_units = tf.placeholder(tf.float32, [None, (graphicSize*graphicSize)])
#-- In this example, the dimensionality is ? x 786
#
#- Define output
output_units = tf.placeholder(tf.float32, [None, numOutputs])
#-- In this example, the dimensionality is ? x 10

# Setting up weights and biases of the model
#- We always have L - 1 weight/bias tensors where L is the number of layers
#- Declare weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([graphicSize*graphicSize, numUnits_hidden1], stddev = 0.03), name = 'W1')
b1 = tf.Variable(tf.random_normal([numUnits_hidden1]), name = 'b1')
#- Declare weights connecting hidden layer to output
W2 = tf.Variable(tf.random_normal([numUnits_hidden1, numOutputs], stddev = 0.03), name = 'W2')
b2 = tf.Variable(tf.random_normal([numOutputs]), name = 'b2')

# Now, how does activation flow through this neural network? We need to define how the layers
# connect to each other. Once we run the following code, tensorflow implicitly creates a graph
# connecting the layers. We have control over what the connections actually mean though (e.g. additive, multiplicative, etc.)
#- First, we calculate the input -> hidden1 connection
hidden_actv = tf.add(tf.matmul(input_units, W1), b1)
hidden_actv = tf.nn.relu(hidden_actv)
#-- Together, these two calls calculate the activation (output) of the input -> hidden connections.
#-- First, the input and hidden layer 1 are multiplied, then the bias is added
#-- Next, the activation is modified by applying the rectified linear unit function to these units
#- Next, we calculate the hidden1 -> output connection
output_actv = tf.nn.softmax(tf.add(tf.matmul(hidden_actv, W2), b2))
#- Following this command, we have the output activity in softmax form

# The basis of machine learning is optimizing over some cost or loss function. We now need to define that
#- In this instance, we will use cross entropy
output_actv_clipped = tf.clip_by_value(output_actv, 1e-10, 0.9999999) # Clips so we never get log(0)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(output_units * tf.log(output_actv_clipped)
                         		+ (1 - output_units) * tf.log(1 - output_actv_clipped), axis=1))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
#- Here's where the optimization occurs -- by minimizing cross entropy error

# Initialize everything
init_op = tf.global_variables_initializer()

# Define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(output_units, 1), tf.argmax(output_actv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#########################
### RUNNING THE MODEL ###
#########################
# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size) # Total number of batches within an epoch based on length of mnist
   for epoch in range(epochs): # We train the model epoch times
        avg_cost = 0
        for i in range(total_batch): # Cycle through the number of batches in each epoch
            batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size) # next_batch is a function with MNIST. Not in all of tensorflow
           	# First line of sess.run(): Runs both optimiser and cross_entropy operations we defined -- uses feed_dict to define input
           	# Second line of sess.run(): Feeding in values to the input and output
            _, c = sess.run([optimiser, cross_entropy],
                         feed_dict={input_units: batch_x, output_units: batch_y})
            avg_cost += c / total_batch # Calculate average cost (to see what the model is doing by printing in next two lines)
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={input_units: mnist.test.images, output_units: mnist.test.labels}))


