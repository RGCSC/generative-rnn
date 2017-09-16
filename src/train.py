import tensorflow as tf
import numpy as np
import itertools
import pandas
import nltk
import random

################## LOAD DATA ##################
NUM_PROBS = 79
solution_mapping = {1:34,2:33,3:50,4:50,5:44,6:50,7:50,8:50,9:50,10:42,
                    11:4,12:50,13:50,14:50,15:50,16:50,17:47,18:27,19:50,20:50,
                    21:50,22:50,23:50,24:50,25:50,26:50,27:50,28:50,29:50,30:50,
                    31:50,32:50,33:50,34:50,35:50,36:50,37:50,38:50,39:50,40:49,
                    41:50,42:50,43:31,44:34,45:49,46:19,47:50,48:50,49:43,50:47,
                    51:31,52:50,53:17,54:50,55:50,56:50,57:50,58:50,59:50,60:50,
                    61:15,62:50,63:30,64:35,65:50,66:11,67:50,68:26,69:50,70:28,
                    71:50,72:46,73:50,74:44,75:34,76:50,77:50,78:50,79:49}

max_n = 0
descs = []
for i in range(NUM_PROBS):
    num = ''
    if i < 10:
         num = '0' + str(i) 
    else:
        num = str(i)
    state = nltk.word_tokenize(open('desc_' + num + '.txt', 'r').read().lower())
    if len(state) > max_n:
      max_n = len(state)
    descs.append(state)

vocab = set(list(' '.join(descs)))

word_indices = dict((s, i) for i, s in enumerate(vocab))
indices_word = dict((i, s) for i, s in enumerate(vocab))

code_mat = []
code_ch = set()
for i in range(NUM_PROBS):
    problem = []
    prob_num=''
    if i < 10:
        prob_num = '0' + str(i)
    else:
        prob_num = str(i)
    for j in range(solution_mapping[i + 1]):
        problem.append(open('code_' + prob_num + '_(' + str(j) + '.txt').read().lower())
    code_mat.append(problem)

code_ch = set(list(''.join(list(itertools.chain.from_iterable(code_mat)))))
code_ch.remove(' ')
print("Total chars: ", len(code_ch))
###############################################



############### PREPROCESS DATA ###############

# 'Convert' chars to indices and indices to chars
char_indices = dict((c, i) for i, c in enumerate(code_ch))
indices_char = dict((i, c) for i, c in enumerate(code_ch))
                  
# cut code
maxlen = 10
step = 4
# x, y
seqs = []
next_chars = []
for row in code_mat:
    temp1 = []
    temp2 = []
    for j in range(0, len(row) - maxlen, step):
        temp1.append(row[j: j + maxlen])
        temp2.append(row[j + maxlen])
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

X_english = np.zeros((NUM_PROBS, max_n, len(vocab)), dtype=np.bool)

X_code = np.zeros((arr.shape[1], maxlen, len(code_ch), NUM_PROBS), dtype=np.bool)
y = np.zeros((arr.shape[1], len(code_ch), NUM_PROBS), dtype=np.bool)

for i in range(NUM_PROBS):
  for pos, word in enumerate(descs[i]):
    X_english[i, pos, word_indices[word]] = 1
      
      
for i in range(len(seqs)):
    for j, seq in enumerate(seqs[i]):
        for pos, char in enumerate(seq):
            X_code[j, pos, char_indices[char], i] = 1
        y[j, char_indices[next_chars[j]], i] = 1
      
###############################################



################# BUILD MODEL #################
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
english_model.add(tf.contrib.keras.layers.Embedding(input_dim=len(vocab),
                                                    output_dim=128))
                  
code_model = tf.contrib.keras.models.Sequential()
code_model.add(tf.contrib.keras.layer.LSTM(128, input_shape=(maxlen, len(code_ch), NUM_PROBS)))
        
model = tf.contrib.keras.models.Sequential()
model.add(tf.contrib.keras.layers.Merge([english_model, code_model]), mode='concat')
model.add(tf.contrib.keras.layers.Dense(128, activation='softmax'))

###############################################



############# RUN AND SAVE MODEL ##############
model.compile(loss='categorial_crossentropy', optimizer=tf.contrib.keras.optimizers.RMSprop(lr=0.01))
model.fit(X_english, [X_code,y_code], batch_size=128,epochs=20,validation_split=0.2)

# train code model
# for iteration in range(1, 60):
#     print()
#     print('-'*50)
#     print('Iteration', iteration)
#     model.fit(X, y, 
#             batch_size=128,
#             epochs=1)

#     problem = random.randint(0, NUM_PROBS-1)
#     start_index = random.randint(0, len)

###############################################
