# -*- coding: utf-8 -*-

__author__ = 'PC-LiNing'

import numpy as np
import gensim

word_embedding_size = 200
pos_embedding_size = 100

model_word = gensim.models.KeyedVectors.load_word2vec_format('../word2vec/kejso_word.bin', binary=True)
model_pos = gensim.models.KeyedVectors.load_word2vec_format('../word2vec/kejso_pos.bin', binary=True)

wordVocab = [k for (k,v) in model_word.vocab.items()]
word_vocab_size = len(wordVocab)
print("word model vocab size: "+str(word_vocab_size))

posVocab = [k for (k,v) in model_pos.vocab.items()]
pos_vocab_size = len(posVocab)
print("pos model vocab size: "+str(pos_vocab_size))


# word embedding 's dimension is 200
def getSentence_matrix(sentence, pos_string, Max_length = 150):
    words=sentence.split()
    poses=pos_string.split()
    sent_matrix=np.zeros(shape=(Max_length,word_embedding_size),dtype=np.float32)
    pos_matrix=np.zeros(shape=(Max_length,pos_embedding_size),dtype=np.float32)
    i=0
    for word in words:
        if word in wordVocab:
            result = model_word[word]
            if result.shape == (200,):
                sent_matrix[i] = result
        i += 1
    i = 0
    for pos in poses:
        if pos in posVocab:
            result = model_pos[pos]
            if result.shape == (100,):
                pos_matrix[i] = result
        i += 1

    sent_matrix_expand = np.expand_dims(sent_matrix, axis=0)
    pos_matrix_expand = np.expand_dims(pos_matrix, axis=0)
    # sentence_embedding = [1,150,300]
    sentence_embedding = np.concatenate((sent_matrix_expand, pos_matrix_expand), axis=2)
    return sentence_embedding


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
    if label == 0:
        return "Cause"
    if label == 1:
        return "Describe"
    if label == 2:
        return "From"
    if label == 3:
        return "Identity"
    if label == 4:
        return "Medicine"
    if label == 5:
        return "Part"
    if label == 6:
        return "Position"
    if label == 7:
        return "Other"
    return "Error"


"""
sentence = "目的 ： 探讨 对 乌头 碱中毒 引起 室 性 心律失常 患者 采用 胺 碘 酮 治疗 的 临床 效果"
pos_string = "n w v p n nhd v n ng gb n v n n g v uj vn n"
embed = getSentence_matrix(sentence, pos_string)
print(embed)
print(embed.shape)
"""