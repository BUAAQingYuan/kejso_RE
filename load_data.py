__author__ = 'PC-LiNing'

import numpy
import codecs
import util_word2vec
import datetime

word_embedding_size = 200
pos_embedding_size = 100
num_classes = 8
final_mebedding_size = word_embedding_size + pos_embedding_size
# train size = 178433
# test size = 9919
MAX_DOCUMENT_LENGTH = 150

"""
# [-49,48]
# PF_embeddings = numpy.random.uniform(low=-0.5,high=0.5,size=(98, PF_dim))
PF_embeddings = numpy.load('PF_embeddings.npy')
# [-49,48]
# n is sentence length , max_length is the max length.
# return [max_length,2*PF_dim]
def  get_PF(e1,e2,n,max_length):
    pos_embeddings = numpy.zeros(shape=(max_length, 2*PF_dim),dtype=numpy.float32)
    vec = range(n)
    pos = 0
    for i in vec:
        p1 = i - e1 + 49
        p2 = i - e2 + 49
        if p1 < 0 or p1 > 97:
            d1 = numpy.zeros(shape=(PF_dim,), dtype=numpy.float32)
        else:
            d1 = PF_embeddings[p1]
        if p2 < 0 or p2 > 97:
            d2 = numpy.zeros(shape=(PF_dim,), dtype=numpy.float32)
        else:
            d2 = PF_embeddings[p2]
        pos_vec = numpy.concatenate((d1,d2),axis=0)
        pos_embeddings[pos] = pos_vec
        pos += 1
    return pos_embeddings
"""


# (sent,seg,type)
def load_corpus_data(data_file):
    file = codecs.open(data_file, encoding='utf-8')
    sentence=[]
    sentence_seg=[]
    label=[]
    pos=[]
    i=1
    for line in file.readlines():
        if i % 3 == 1:
            items = line.replace('\n','').split("\t")
            sentence.append(items[1])
            label.append(items[0])
        if i % 3 == 2:
            sentence_seg.append(line.replace('\n',''))
        if i % 3 == 0:
            pos.append(line.replace('\n',''))
        i+=1
    # parse
    train_data=[]
    for i in range(0, len(sentence)):
        sen=sentence_seg[i]
        type=label[i]
        seg = pos[i]
        words = sen.split(" ")
        poses = seg.split(" ")
        if len(words) == len(poses) and len(words) <= 150:
            train_data.append((sen,seg,type))
    max_sent_length = max([len(item[0].split(" ")) for item in train_data])
    max_pos_length = max([len(item[1].split(" ")) for item in train_data])
    print("example size: " + str(len(train_data)))
    print("max_sent_length: " + str(max_sent_length))
    print("max_pos_length: " + str(max_pos_length))
    return train_data


# relation label to number
# 0 Cause
# 1 Describe
# 2 From
# 3 Identity
# 4 Medicine
# 5 Part
# 6 Position
# 7 Other
def transfer_label(label):
    if label.startswith('Cause'):
        return 0
    if label.startswith('Describe'):
        return 1
    if label.startswith('From'):
        return 2
    if label.startswith('Identity'):
        return 3
    if label.startswith('Medicine'):
        return 4
    if label.startswith('Part'):
        return 5
    if label.startswith('Position'):
        return 6
    if label.startswith('Other'):
        return 7


# parse kejso train data
# (sent,seg,type)
def load_train_data():
    semeval_data = load_corpus_data("data/train.txt")
    Train_Size = len(semeval_data)
    train_word = numpy.ndarray(shape=(Train_Size, MAX_DOCUMENT_LENGTH, word_embedding_size),dtype=numpy.float32)
    train_pos = numpy.ndarray(shape=(Train_Size,MAX_DOCUMENT_LENGTH, pos_embedding_size),dtype=numpy.float32)
    train_labels = []
    i = 0
    for one in semeval_data:
        sentence = one[0]
        pos_string = one[1]
        label = one[2]
        train_labels.append(transfer_label(label))
        train_word[i],train_pos[i] = util_word2vec.getSentence_matrix(sentence, pos_string, MAX_DOCUMENT_LENGTH)
        i+=1
        if i % 1000 == 0:
            time_str = datetime.datetime.now().isoformat()
            print("{}: process {:g}%".format(time_str, i*100.0/Train_Size))
    train_labels = numpy.asarray(train_labels, dtype=numpy.float32)
    return train_word, train_pos, train_labels


# parse SemEval test data
def load_test_data():
    semeval_data = load_corpus_data("data/test.txt")
    Test_Size = len(semeval_data)
    test_word = numpy.ndarray(shape=(Test_Size, MAX_DOCUMENT_LENGTH, word_embedding_size),dtype=numpy.float32)
    test_pos = numpy.ndarray(shape=(Test_Size,MAX_DOCUMENT_LENGTH, pos_embedding_size),dtype=numpy.float32)
    test_labels = []
    i = 0
    for one in semeval_data:
        sentence = one[0]
        pos_string = one[1]
        label = one[2]
        test_labels.append(transfer_label(label))
        test_word[i],test_pos[i] = util_word2vec.getSentence_matrix(sentence, pos_string, MAX_DOCUMENT_LENGTH)
        i+=1
        if i % 1000 == 0:
            time_str = datetime.datetime.now().isoformat()
            print("{}: process {:g}%".format(time_str, i*100.0/Test_Size))
    test_labels = numpy.asarray(test_labels, dtype=numpy.float32)
    return test_word, test_pos, test_labels


print("get train data ...")
train_sent, train_pos, train_labels = load_train_data()
print(train_sent.shape)
print(train_pos.shape)
print(train_labels)
numpy.save('train_word.npy', train_sent)
numpy.save('train_pos.npy', train_pos)
numpy.save('train_label.npy', train_labels)

