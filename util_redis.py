__author__ = 'PC-LiNing'

import numpy as np
import redis

word_embedding_size = 200
pos_embedding_size = 100
# redis
r_word = redis.StrictRedis(host='10.2.4.78', port=6379, db=0)
r_pos = redis.StrictRedis(host='10.2.4.78', port=6379, db=1)


# word embedding 's dimension is 200
def  getSentence_matrix(sentence,pos_string,Max_length):
     words=sentence.split()
     poses=pos_string.split()
     sent_matrix=np.zeros(shape=(Max_length,word_embedding_size),dtype=np.float32)
     pos_matrix=np.zeros(shape=(Max_length,pos_embedding_size),dtype=np.float32)
     i=0
     for word in words:
         result = r_word.get(word)
         if result is not None:
             vec = np.fromstring(result, dtype=np.float32)
             if vec.shape == (200,):
                 sent_matrix[i]=vec
         i+=1
     i=0
     for pos in poses:
         result = r_pos.get(pos)
         if result is not None:
             vec = np.fromstring(result, dtype=np.float32)
             if vec.shape == (100,):
                 pos_matrix[i]=vec
         i+=1
     return sent_matrix, pos_matrix

