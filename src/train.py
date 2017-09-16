import tensorflow as tf
import pandas
import numpy as np
import nltk

########### LOAD DATA ##############
desc = open('description.txt', 'r').read().lower()
# Change constant below
NUM_CODES = 10
code_mat = []
for i in range(1, NUM_CODES+1):
  code_mat.append(open('code'+i+'.txt', 'r').read.lower())
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

desc_ch = set(list(desc))
code_ch = set(list(code))
desc_ch.remove(' ')
code_ch.remove(' ')
print('total chars:', len(desc_ch) + len(code_ch))
# 'Convert' chars to indices and indices to chars
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
                  
# cut code
maxlen = 10
step = 4
# x, y
seqs = []
next_chars = []
for row in code_mat:
    temp1 = []
    temp2 = []
    for j in range(len(row) - maxlen, step):
        temp1.append(row[j: j+maxlen])
        temp2.append(row[j+maxlen])
    seqs.append(temp1)
    next_chars.append(temp2)
print('num seqs:', len(seqs))

# Vectorization of inputs, outputs
max_row = 0
for row in seqs:
    if len(row) > max_row:
        max_row = len(row)

for row in seqs:
    for i in range(max_row - len(row)):
        row.append(' ')
arr = np.array(seqs)
X = np.zeros((arr.shape[1], maxlen, len(code_ch), NUM_CODES), dtype=np.bool)
y = np.zeros((arr.shape[1], len(code_ch), NUM_CODES), dtype=np.bool)

for i in range(len(seqs)):
    for j, seq in enumerate(seqs[i]):
        for pos, char in enumerate(seq):
            X[j, pos, char_indices[char], i] = 1
        y[j, char_indices[next_chars[j]], i] = 1

      
####################################



########### BUILD MODEL ###########
# Sample character from probability distribution
def sample(preds, temperature=1.0):
    # Convert
    preds = np.asarray(preds).astype('float64')
    # Softmax
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

english_model = tf.contrib.keras.models.Sequential()
english_model.add(tf.contrib.keras.layers.Embedding(input_dim=english_vs,
                                                    output_dim=32))
english_model.add(tf.contrib.keras.layer.LSTM(32))
                  
code_model = tf.contrib.keras.models.Sequential()
code_model.add(tf.contrib.keras.layer.LSTM(128, input_shape=(maxlen, len(code_ch), NUM_CODES)))
        
model = tf.contrib.keras.models.Sequential()
model.add(tf.contrib.keras.layers.Merge([english_model, code_model]), mode='concat')
model.add(tf.contrib.keras.layers.Dense(128, activation = 'softmax'))
####################################



######### RUN MODEL, SAVE ########
model.compile(loss='categorial_crossentropy', optimizer=tf.contrib.keras.optimizers.RMSprop(lr=0.01))



#################################


