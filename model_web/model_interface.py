import tensorflow as tf
from ops import conv2d
import numpy
import math


max_document_length = 150
NUM_CLASSES = 8
EMBEDDING_SIZE = 200
POS_SZIE = 100
FINAL_EMBEDDING_SIZE = EMBEDDING_SIZE + POS_SZIE
start_learning_rate = 1e-3

# input is sentence
train_data_node = tf.placeholder(tf.float32,shape=(None,max_document_length,FINAL_EMBEDDING_SIZE))

# train_labels_node = tf.placeholder(tf.float32,shape=(None,NUM_CLASSES))

dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")


filter_sizes = [2,3,4,5]
filter_numbers = [300,200,100,50]
# input attention matrix
d_c = sum(filter_numbers)
# class embeddings matrix
init_class = math.sqrt(6.0 / (d_c + NUM_CLASSES))
fc_weights = tf.Variable(tf.random_uniform([d_c, NUM_CLASSES],
                                                minval=-init_class,
                                                maxval=init_class,
                                                dtype=tf.float32))
fc_biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES], dtype=tf.float32))


# model
# data = [batch_size,n,embed]
def model(data):
    # exp_data = [batch_size,n,embed,1]
    exp_data = tf.expand_dims(data, axis=-1)
    pooled_outputs = []
    for idx, filter_size in enumerate(filter_sizes):
        conv = conv2d(exp_data,filter_numbers[idx],filter_size,FINAL_EMBEDDING_SIZE,name="kernel%d" % idx)
        # 1-max pooling,leave a tensor of shape[batch_size,1,1,num_filters]
        pool = tf.nn.max_pool(conv,ksize=[1,max_document_length-filter_size+1,1,1],strides=[1, 1, 1, 1],padding='VALID')
        pooled_outputs.append(tf.expand_dims(tf.squeeze(pool), axis=0))

        if len(filter_sizes) > 1:
            cnn_output = tf.concat(pooled_outputs, axis=1)
        else:
            cnn_output = pooled_outputs[0]

    # add dropout
    # dropout_output = [batch_size, num_filter]
    dropout_output = tf.nn.dropout(cnn_output, dropout_keep_prob)
    # fc layer
    fc_output = tf.matmul(dropout_output, fc_weights) + fc_biases
    return fc_output


# Training computation
logits = tf.nn.softmax(model(train_data_node))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_node,
#                                                              logits=tf.clip_by_value(logits,1e-10,1.0)))
# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(fc_weights) + tf.nn.l2_loss(fc_biases))
# loss += 0.05 * regularizers
# tf.summary.scalar('loss', loss)

# optimizer
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_rate = tf.Variable(start_learning_rate,name="learning_rate")
# learning_rate=tf.train.exponential_decay(start_learning_rate,global_step*BATCH_SIZE,train_size,0.9,staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
# grads_and_vars = optimizer.compute_gradients(loss)
# train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Evaluate model
train_predict = tf.argmax(logits,1)
# train_label = tf.argmax(train_labels_node,1)
# train accuracy
# train_correct_pred = tf.equal(train_predict,train_label)
# train_accuracy = tf.reduce_mean(tf.cast(train_correct_pred, tf.float32))
# tf.summary.scalar('acc', train_accuracy)

# run the testing
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "model.ckpt")

'''
test_word = numpy.load("test_word.npy")
test_pos = numpy.load("test_pos.npy")
test_labels = numpy.load("test_dis_label.npy")
print(test_word.shape)
print(test_pos.shape)
print(test_labels.shape)
# concatenation
x_test = numpy.concatenate((test_word, test_pos), axis=2)
y_test = test_labels
print(x_test.shape)
'''


def predict(x_batch):
    feed_dict = {train_data_node: x_batch, dropout_keep_prob: 1.0}
    scores, y_predict = sess.run([logits, train_predict], feed_dict=feed_dict)
    return scores, y_predict



