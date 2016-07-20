# Deep Autoencoder to learn to discriminate exponential heating behaviour of tissue from background activity
# Author: nih23

import json
import h5py
import scipy as sp
import numpy as np
#get_ipython().magic(u'matplotlib notebook')
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l1,l2,l1l2, activity_l2
from keras.models import model_from_json

### CONSTANTS
nb_classes = 2
classificationHiddenSz = 10
windowSz = 300
hiddenSz = 25 
noEpochs = 100 
pData = "irr_dr_training.mat"
pModelWeights = "model_weights_ae"
pModelConfig = "model_config_ae"

### READ DATA
f = h5py.File(pData,"r")
bg = f["seqBackground"]
fg = f["seqSpuelung"]

x_c1 = np.array(fg.value)
x_c2 = np.array(bg.value)

f.close()

x_fg = np.transpose(x_c1[601:901,0:9402])
x_bg = np.transpose(x_c2[601:901,0:9402])

print(x_fg.shape)
print(x_bg.shape)



### PREPROCESS DATA
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

X_train = np.concatenate((x_fg4[0:noFgSamples , :], x_bg4[0:noBgSamples , :]), axis = 0)
Y_train = np.concatenate((y_fg[0:noFgSamples], y_bg[0:noBgSamples]), axis = 0)
X_test = np.concatenate((x_fg4[noFgSamples: , :], x_bg4[noBgSamples: , :]), axis = 0)
Y_test = np.concatenate((y_fg[noFgSamples:], y_bg[noBgSamples:]), axis = 0)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#Y_train = np_utils.to_categorical(Y_train, nb_classes)
#Y_test = np_utils.to_categorical(Y_test, nb_classes)

####################
### INITIALIZE MODEL
####################
#regulPen = l2(0.001)
regulPen = l1l2(l1=1, l2=0.01)
act = 'sigmoid'

input_data = Input(shape=(windowSz,))
encoded = Dense(hiddenSz, W_regularizer=regulPen, activation=act, name="encoder")(input_data)
dropout1 = Dropout(0.1)(encoded)
encoded2 = Dense(hiddenSz/2, W_regularizer=regulPen, activation=act, name="encoder2")(dropout1)
dropout2 = Dropout(0.1)(encoded2)
decoded2 = Dense(hiddenSz, W_regularizer=regulPen, activation=act, name="decoder2")(encoded2)
decoded = Dense(windowSz, W_regularizer=regulPen, activation=act, name="decoder")(decoded2)
#autoencoder = Model(input=input_data, output=decoded)
#encoder = Model(input=input_data, output=encoded)

#encoded_input = Input(shape=(hiddenSz,))
#decoder_layer = autoencoder.layers[-1]
#decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
#autoencoder.compile(optimizer='rmsprop', loss='mse')

#autoencoder.fit(X_train, X_train, nb_epoch=10, batch_size=int(round(windowSz/2)), shuffle=True, validation_data=(X_test, X_test))

#X_test_encoded = encoder.predict(x_bg4)
#X_test_decoded = decoder.predict(X_test_encoded)

#plt.plot(X_test_decoded[0,:])
#plt.plot(x_bg4[0,:])
#plt.show()


############
### TRAINING
############

from keras.layers.core import Activation
#Fine-tuning
print('Fine-tuning')
sm = Dense(classificationHiddenSz, activation='sigmoid', name="classifier_intermediate")(dropout2)
sm2 = Dense(1, activation='sigmoid', name="classifier")(sm)

model = Model(input = input_data, output = [sm2, decoded ])

model.compile(optimizer='sgd', metrics={'classifier': 'accuracy'}, 
loss={'classifier': 'binary_crossentropy', 'decoder': 'mse'},
#loss_weights={'classifier': 1., 'decoder': 0.01}
)
model.fit(X_train, [Y_train, X_train ], batch_size=32, nb_epoch=noEpochs,
          validation_data=(X_test, [Y_test, X_test]))
score = model.evaluate(X_test, [Y_test, X_test], verbose=1)
print('')
print('Test score 1 ', score[0])
print('Test score 2 ', score[1])
print('Test acc 1 ', score[2])
print('Test acc 2 ', score[3])
json_rep = model.to_json()
model.save_weights(pModelWeights, overwrite=True)
f = open(pModelConfig,"wb")
json.dump(json_rep,f)
