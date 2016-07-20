# Deep Autoencoder to learn to discriminate exponential heating behaviour of tissue from background activity
# Author: nih23 

import h5py
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, merge
from keras.models import Model
from keras.regularizers import l1,l2,l1l2, activity_l2

#############
### CONSTANTS
#############
nb_classes = 2
windowSz = 300
hiddenSz = 90
epochs = 100

#############
### READ DATA
#############
f = h5py.File("irr_dr_training.mat","r")
bg = f["seqBackground"]
fg = f["seqSpuelung"]
x_c1 = np.array(fg.value)
x_c2 = np.array(bg.value)
f.close()
x_fg = np.transpose(x_c1[601:901,0:9401])
x_bg = np.transpose(x_c2[601:901,0:9401])
print(x_fg.shape)
print(x_bg.shape)

###################
### PREPROCESS DATA
###################
(nX1,nT1) = x_fg.shape
x_fg2 = np.split(x_fg,nT1/windowSz,1)
x_fg3 = np.asarray(x_fg2)
print(x_fg3.shape)
(dx1,dx2,dx3) = x_fg3.shape
x_fg4 = np.reshape(x_fg3,(dx1*dx2,dx3))
noFgSamples = x_fg4.shape
y_fg = np.ones(noFgSamples[0])
noFgSamples = int(np.round(0.5 * noFgSamples[0] ))



(nX2,nT2) = x_bg.shape
x_bg2 = np.split(x_bg,nT2/windowSz,1)
x_bg3 = np.asarray(x_bg2)
(dx1,dx2,dx3) = x_bg3.shape
print(x_bg3.shape)
x_bg4 = np.reshape(x_bg3,(dx1*dx2,dx3))
noBgSamples = x_bg4.shape
y_bg = 0*np.ones(noBgSamples[0])
noBgSamples = int(np.round(0.5 * noBgSamples[0]))

#x_bg4 = sp.stats.zscore(x_bg4, axis=1)
#x_fg4 = sp.stats.zscore(x_fg4, axis=1)

bgmin = np.amin(x_bg4, axis=1)
fgmin = np.amin(x_fg4, axis=1)
x_bg4 = ( x_bg4 - bgmin[:,None] ) 
x_fg4 = ( x_fg4 - fgmin[:,None] )
bgmax = np.amax(x_bg4, axis=1)
fgmax = np.amax(x_fg4, axis=1)
x_fg4 = x_fg4 / fgmax[:, None]
x_bg4 = x_bg4 / bgmax[:, None]

print(x_fg4.shape)
print(x_bg4.shape)

X_train_fg = x_fg4[0:noFgSamples , :];
X_train_bg = x_bg4[0:noBgSamples , :];
X_test_fg = x_fg4[noFgSamples: , :];
X_test_bg = x_bg4[noBgSamples: , :];

X_train_fgbg = np.concatenate((x_fg4[0:noFgSamples , :], x_bg4[0:noBgSamples , :]), axis = 0)
Y_train_fgbg = np.concatenate((y_fg[0:noFgSamples], y_bg[0:noBgSamples]), axis = 0)
X_test_fgbg = np.concatenate((x_fg4[noFgSamples: , :], x_bg4[noBgSamples: , :]), axis = 0)
Y_test_fgbg = np.concatenate((y_fg[noFgSamples:], y_bg[noBgSamples:]), axis = 0)

print(X_train_fgbg.shape[0], 'train samples')
print(X_test_fgbg.shape[0], 'test samples')

#Y_train = np_utils.to_categorical(Y_train, nb_classes)
#Y_test = np_utils.to_categorical(Y_test, nb_classes)

####################
### INITIALIZE MODEL
####################

#regulPen = l2(0.001)
## we prefer L1 and L2 penalties on our model to enforce sparsity of our learnt parameters
regulPen = l1l2(l1=1, l2=0.01)
act = 'sigmoid'

### STACKED AUTOENCODER ON FOREGROUND SIGNAL
input_data = Input(shape=(windowSz,))
encoded_fg = Dense(hiddenSz, W_regularizer=regulPen, activation=act, name="encoder_fg")(input_data)
dropout1_fg = Dropout(0.5)(encoded_fg)
encoded2_fg = Dense(hiddenSz/10, W_regularizer=regulPen, activation=act, name="encoder2_fg")(dropout1_fg)
dropout2_fg = Dropout(0.5)(encoded2_fg)
decoded2_fg = Dense(hiddenSz, W_regularizer=regulPen, activation=act, name="decoder2_fg")(encoded2_fg)
decoded_fg = Dense(windowSz, W_regularizer=regulPen, activation=act, name="decoder_fg")(decoded2_fg)


### STACKED AUTOENCODER ON BACKGROUND SIGNAL
encoded_bg = Dense(hiddenSz, W_regularizer=regulPen, activation=act, name="encoder_bg")(input_data)
dropout1_bg = Dropout(0.5)(encoded_bg)
encoded2_bg = Dense(hiddenSz/10, W_regularizer=regulPen, activation=act, name="encoder2_bg")(dropout1_bg)
dropout2_bg = Dropout(0.5)(encoded2_bg)
decoded2_bg = Dense(hiddenSz, W_regularizer=regulPen, activation=act, name="decoder2_bg")(encoded2_bg)
decoded_bg = Dense(windowSz, W_regularizer=regulPen, activation=act, name="decoder_bg")(decoded2_bg)

############
### TRAINING
############

from keras.layers.core import Activation
#Fine-tuning
print('Training phase')


################################################################################################################################################################################################################################################################################################

############ TODO: FIX MODEL DEFINITION! THIS ONE SEEMS DEFINITELY WRONG!! (we dont want to plug encoded data into the fitting function.. the model should do this itself ...)

################################################################################################################################################################################################################################################################################################

## merge autoencoders layer
merged_ae = merge([encoded2_fg, encoded2_bg], mode='concat')

## final classifier
sm = Dense(50, activation='sigmoid', name="classifier_intermediate")(merged_ae)
sm2 = Dense(1, activation='sigmoid', name="classifier")(sm)

model = Model(input = [input_data], output = [sm2, encoded2_fg, encoded2_bg])

model.compile(optimizer='sgd', metrics={'classifier': 'accuracy'}, 
loss={'classifier': 'binary_crossentropy', 'encoder2_fg': 'mse', 'encoder2_bg': 'mse'},
#loss_weights={'classifier': 1., 'decoder': 0.01}
)

## merge input data
merged_input_training_fg = np.concatenate((X_train_fg, X_train_fg), axis = 1)
merged_input_training_bg = np.concatenate((X_train_bg, X_train_bg), axis = 1)
merged_input_training_fgbg = np.concatenate((merged_input_training_fg, merged_input_training_bg), axis = 0)

## merge target data of first and second autoencoder
somezeros = 0*np.ones([noBgSamples, windowSz])
merged_ae_input_training_fg = np.concatenate((X_train_fg, somezeros), axis = 0)
merged_ae_input_training_bg = np.concatenate((somezeros, X_train_bg), axis = 0)



## fit model
model.fit(merged_input_training_fgbg, [Y_train_fgbg, merged_ae_input_training_fg, merged_ae_input_training_bg ], batch_size=32, nb_epoch=10,
          validation_data=([X_test_fg, X_test_bg], [Y_test_fgbg, X_test_fg, X_test_bg]))

merged_input_testing = np.concatenate((X_test_fgbg, X_test_fgbg), axis = 1)


score = model.evaluate(merged_input_testing, [Y_test_fgbg, X_test_fgbg, X_test_fgbg], verbose=1)
print('')
print('Test score 1 ', score[0])
print('Test score 2 ', score[1])
print('Test acc 1 ', score[2])
print('Test acc 2 ', score[3])
