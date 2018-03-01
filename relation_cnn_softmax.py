__author__ = 'PC-LiNing'

import datetime
import numpy
import tensorflow as tf
import data_helpers
from ops import conv2d
import argparse
from sklearn.metrics import recall_score,accuracy_score,f1_score
import math


NUM_CLASSES = 8
EMBEDDING_SIZE = 200
POS_SZIE = 100
FINAL_EMBEDDING_SIZE = EMBEDDING_SIZE + POS_SZIE
NUM_CHANNELS = 1
SEED = 66478
BATCH_SIZE = 512
NUM_EPOCHS = 20
EVAL_FREQUENCY = 100
META_FREQUENCY = 100
# learning rate
learning_rate_decay = 0.5
start_learning_rate = 1e-3
decay_delta = 0.005
min_learning_rate = 5e-5
# train
steps_each_check = 500
max_document_length = 150

# test size
Test_Size = 9919
Train_Size = 178433

# FLAGS=tf.app.flags.FLAGS
FLAGS = None


def train(argv=None):
    # load data
    print("Loading data ... ")
    train_word = numpy.load("train_word.npy")
    train_pos = numpy.load("train_pos.npy")
    train_labels = numpy.load("train_dis_label.npy")
    print(train_word.shape)
    print(train_pos.shape)
    print(train_labels.shape)
    test_word = numpy.load("test_word.npy")
    test_pos = numpy.load("test_pos.npy")
    test_labels = numpy.load("test_dis_label.npy")
    print(test_word.shape)
    print(test_pos.shape)
    print(test_labels.shape)
    # concatenation
    x_train = numpy.concatenate((train_word, train_pos), axis=2)
    y_train = train_labels
    x_test = numpy.concatenate((test_word, test_pos), axis=2)
    y_test = test_labels
    print(x_train.shape)
    print(x_test.shape)

    # Test record
    record_f1 = []
    record_acc = []
    # record_predicts = []
    # record_alpha = []

    filter_sizes = [2,3,4,5]
    filter_numbers = [300,200,100,50]
    # input
    # input is sentence
    train_data_node = tf.placeholder(tf.float32,shape=(None,max_document_length,FINAL_EMBEDDING_SIZE))

    train_labels_node = tf.placeholder(tf.float32,shape=(None,NUM_CLASSES))

    dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

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
            pooled_outputs.append(tf.squeeze(pool))

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
    logits = model(train_data_node)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_node, logits=tf.clip_by_value(logits,1e-10,1.0)))
    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc_weights) + tf.nn.l2_loss(fc_biases))
    loss += 0.05 * regularizers
    tf.summary.scalar('loss', loss)

    # optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    learning_rate = tf.Variable(start_learning_rate,name="learning_rate")
    # learning_rate=tf.train.exponential_decay(start_learning_rate,global_step*BATCH_SIZE,train_size,0.9,staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Evaluate model
    train_predict = tf.argmax(logits,1)
    train_label = tf.argmax(train_labels_node,1)
    # train accuracy
    train_correct_pred = tf.equal(train_predict,train_label)
    train_accuracy = tf.reduce_mean(tf.cast(train_correct_pred, tf.float32))
    tf.summary.scalar('acc', train_accuracy)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    def compute_index(y_label,y_predict):
        # macro
        macro_acc = accuracy_score(y_label,y_predict)
        macro_f1 = f1_score(y_label,y_predict, average='macro')
        print("{}: acc {:g}, recall {:g}, f1 {:g} ".format("macro", macro_acc,
                                                           recall_score(y_label, y_predict, average='macro'),
                                                           macro_f1))
        # macro
        print("{}: acc {:g}, recall {:g}, f1 {:g} ".format("micro",accuracy_score(y_label,y_predict),
                                                           recall_score(y_label, y_predict, average='micro'),
                                                           f1_score(y_label,y_predict, average='micro')))

        # weighted
        print("{}: acc {:g}, recall {:g}, f1 {:g} ".format("weighted",accuracy_score(y_label,y_predict),
                                                           recall_score(y_label, y_predict, average='weighted'),
                                                           f1_score(y_label,y_predict, average='weighted')))
        return macro_acc, macro_f1

    def dev_step(x_batch,y_batch,best_test_loss,sess):
        feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,dropout_keep_prob:1.0}
        # Run the graph and fetch some of the nodes.
        # test dont apply train_op (train_op is update gradient).
        summary,step, losses, lr,acc,y_label,y_predict= sess.run([merged,global_step, loss,learning_rate,train_accuracy,train_label,train_predict]
                                                                   ,feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, lr {:g} ,acc {:g}".format(time_str, step, losses,lr,acc))
        # print("{}: step {}, loss {:g} ,acc {:g}".format(time_str, step, losses,acc))
        # compute index
        current_acc, current_f1 = compute_index(y_label, y_predict)
        # put in record
        record_f1.append(current_f1)
        record_acc.append(current_acc)
        new_best_test_loss = best_test_loss
        # decide if need to decay learning rate
        if (step % steps_each_check < 100) and (step > 100):
            loss_delta = (best_test_loss if best_test_loss is not None else 0 ) - losses
            if best_test_loss is not None and loss_delta < decay_delta:
                print('validation loss did not improve enough, decay learning rate')
                current_learning_rate = min_learning_rate if lr * learning_rate_decay < min_learning_rate else lr * learning_rate_decay
                if current_learning_rate == min_learning_rate:
                    print('It is already the smallest learning rate.')
                sess.run(learning_rate.assign(current_learning_rate))
                print('new learning rate is: ', current_learning_rate)
            else:
                # update
                new_best_test_loss = losses

        return new_best_test_loss

    # run the training
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
        tf.initialize_all_variables().run()
        print('Initialized!')
        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train,y_train)),BATCH_SIZE,NUM_EPOCHS)
        # batch count
        batch_count = 0
        best_test_loss = None
        # Training loop.For each batch...
        for batch in batches:
            batch_count += 1
            if batch_count % EVAL_FREQUENCY == 0:
                print("\nEvaluation:")
                best_test_loss=dev_step(x_test,y_test,best_test_loss,sess)
                print("")
            else:
                if  batch_count % META_FREQUENCY == 99:
                    x_batch, y_batch = zip(*batch)
                    feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,dropout_keep_prob:0.5}
                    # Run the graph and fetch some of the nodes.
                    # option
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _,summary, step, losses, acc = sess.run([train_op,merged,global_step, loss,train_accuracy],
                                                            feed_dict=feed_dict,
                                                            options=run_options,
                                                            run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    train_writer.add_summary(summary, step)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g},acc {:g}".format(time_str, step, losses,acc))
                else:
                    x_batch, y_batch = zip(*batch)
                    feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,dropout_keep_prob:0.5}
                    # Run the graph and fetch some of the nodes.
                    _, summary, step, losses, acc = sess.run([train_op,merged,global_step, loss,train_accuracy],feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, losses,acc))

        # save model
        save_path = saver.save(sess, "model.ckpt")
        print("Model saved in file: ", save_path)
        # save test predicts
        # numpy.save('test_predicts.npy', record_predicts[0])
        # save alpha
        # numpy.save('sdp_att_alpha.npy', record_alpha[0])
        # print best result
        best_acc = max(record_acc)
        best_f1 = max([record_f1[i] for i,a in enumerate(record_acc) if a==best_acc])
        print('best acc: '+str(best_acc)+' '+'F1: '+str(best_f1))
        print('Highest F1: '+str(max(record_f1)))
        train_writer.close()
        test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='/Users/lining/temp/cnn_logs',help='Summaries directory')
    FLAGS = parser.parse_args()
    tf.app.run()