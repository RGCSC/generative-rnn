import tensorflow as tf
import pandas
import numpy as np
import nltk

########### LOAD DATA ##############
desc = open('description.txt', 'r').read()
code1 = open('10211792.txt', 'r').read()
code2 = open('10215386.txt', 'r').read()
code3 = open('10218855.txt', 'r').read()
code4 = open('10224196.txt', 'r').read()
code5 = open('10230779.txt', 'r').read()
code6 = open('10247938.txt', 'r').read()
code7 = open('10251251.txt', 'r').read()
####################################



########## PREPROCESS DATA ##########
#desc_t = nltk.word_tokenize(desc)
#code1_t = nltk.word_tokenize(code1)
#code2_t = nltk.word_tokenize(code2)
#code3_t = nltk.word_tokenize(code3)
#code4_t = nltk.word_tokenize(code4)
#code5_t = nltk.word_tokenize(code5)
#code6_t = nltk.word_tokenize(code6)
#code7_t = nltk.word_tokenize(code7)

code = code1 + code2 + code3 + code4 + code5 + code6 + code7

desc_ch = set(list(desc.lower()))
code_ch = set(list(code.lower()))
desc_ch.remove(' ')
code_ch.remove(' ')
####################################



########### BUILD MODEL ###########
english_model = tf.contrib.keras.models.Sequential()
english_model.add(tf.contrib.keras.layers.Embedding(input_dim=english_vs,
                                                    output_dim=32))
english_model.add(tf.contrib.keras.layer.LSTM(32))
                  
code_model = tf.contrib.keras.models.Sequential()
code_model.add(tf.contrib.keras.layers.Embedding(input_dim=code_vs,
                                                 output_dim=32))
code_model.add(tf.contrib.keras.layer.LSTM(32))
        
model = tf.contrib.keras.models.Sequential()
model.add(tf.contrib.keras.layers.Merge([english_model, code_model]), mode='concat')
model.add(tf.contrib.keras.layers.Dense(128, activation = 'softmax'))
####################################



######### RUN MODEL, SAVE ########
model.compile(loss='categorial_crossentropy', optimizer='adagrad')



#################################


