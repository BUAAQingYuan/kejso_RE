# -*- coding: utf-8 -*-

__author__ = 'PC-LiNing'

import numpy as np
import gensim

word_embedding_size = 200
pos_embedding_size = 100

model_word = gensim.models.KeyedVectors.load_word2vec_format('word2vec/kejso_word.bin', binary=True)
model_pos = gensim.models.KeyedVectors.load_word2vec_format('word2vec/kejso_pos.bin', binary=True)

wordVocab = [k for (k,v) in model_word.vocab.items()]
word_vocab_size = len(wordVocab)
print("word model vocab size: "+str(word_vocab_size))

posVocab = [k for (k,v) in model_pos.vocab.items()]
pos_vocab_size = len(posVocab)
print("pos model vocab size: "+str(pos_vocab_size))


# word embedding 's dimension is 200
def  getSentence_matrix(sentence,pos_string,Max_length):
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
         i+=1
     i=0
     for pos in poses:
         if pos in posVocab:
             result = model_pos[pos]
             if result.shape == (100,):
                pos_matrix[i] = result
         i+=1
     return sent_matrix, pos_matrix

