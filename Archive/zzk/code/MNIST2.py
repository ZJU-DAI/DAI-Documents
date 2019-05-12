import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODES_NUM = 784
LAYER1_NODES_NUM = 500

OUTPUT_NODES_NUM = 10
batch_size = 128

MOVING_AVERAGE_DECAY = 0.99

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, INPUT_NODES_NUM], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODES_NUM], name='y-input')

w1 = tf.Variable(tf.truncated_normal([INPUT_NODES_NUM, LAYER1_NODES_NUM], stddev=0.1),name='w1')
bias1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODES_NUM]))

w2 = tf.Variable(tf.truncated_normal([LAYER1_NODES_NUM,OUTPUT_NODES_NUM],stddev=0.1),name='w2')
bias2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODES_NUM]))

global_step = tf.Variable(0,trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
variable_averages_op = variable_averages.apply(tf.trainable_variables())
layer1_average = tf.nn.relu(tf.matmul(x,variable_averages.average(w1))+variable_averages.average(bias1))
y_average = tf.nn.relu(tf.matmul(layer1_average,variable_averages.average(w2)+variable_averages.average(bias2)))

layer1 = tf.nn.relu(tf.matmul(x, w1) + bias1)
y = tf.nn.relu(tf.matmul(layer1,w2) + bias2)

y = tf.nn.softmax(y)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-8,1)),reduction_indices=[1]))
#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_)

loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

train_op = tf.group(train_step,variable_averages_op)
#with tf.control_dependencies([train_step,variable_averages_op]):
#	train_op = tf.no_op(name='train')

####validation
correct_prediction  = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

correct_prediction_average  = tf.equal(tf.argmax(y_average,1),tf.argmax(y_,1))
accuracy_average = tf.reduce_mean(tf.cast(correct_prediction_average,tf.float32))

#print(tf.trainable_variables())
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    #print('before training w1\n',sess.run(w1[0:10]))

    for i in range(10000):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        loss_value,_ = sess.run([loss,train_op], feed_dict={x: x_batch, y_: y_batch})
        #print('loss:', loss_value)
        print('loss:',sess.run(loss, feed_dict={x: x_batch, y_: y_batch}))

    #print('after training w1\n',sess.run(w1[0:10]))
    print('accuracy = ',sess.run(accuracy,feed_dict={x:mnist.validation.images,y_:mnist.validation.labels}))
    print('accuracy(average) = ',sess.run(accuracy_average,feed_dict={x:mnist.validation.images,y_:mnist.validation.labels}))
    
