import json
import h5py
import numpy as np
#get_ipython().magic(u'matplotlib notebook')
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense, Activation
from keras.layers import Input
from keras.layers.recurrent import LSTM


#############
### CONSTANTS
#############
nb_classes = 2
windowSz = 300
doubleExpModelOutputDim = 4
hiddenSz = 90
noEpochs = 100
pData = "irr_dr_training.mat"
pDoubleExpModel = "model_heatingParamClassifier"


#############
### Load Model for classifying double exponential model parameters
#############
print('-> importing previously trained model')
pModelWeights = "model_weights_ae"
pModelConfig = "model_config_ae"
f = open(pModelConfig,"r")
model_def = json.load(f)

model_heatingParamClassifier = model_from_json(model_def)
### TODO: remove decoder layer ...
model_heatingParamClassifier.load_weights(pModelWeights)
model_heatingParamClassifier.compile(loss='mse', optimizer='adam')
model_heatingParamClassifier.summary()

#############
### READ DATA
#############
print('-> reading data')
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

###################
### PARTITION DATA
###################
# foreground samples
(nX1,nT1) = x_fg.shape
x_fg2 = np.split(x_fg,nT1/windowSz,1)
x_fg3 = np.asarray(x_fg2)
print(x_fg3.shape)
(dx1,dx2,dx3) = x_fg3.shape
x_fg4 = np.reshape(x_fg3,(dx1*dx2,dx3))
noFgSamples = x_fg4.shape
y_fg = np.ones(noFgSamples[0])
noFgSamples = int(np.round(0.5 * noFgSamples[0] ))
# background samples
(nX2,nT2) = x_bg.shape
x_bg2 = np.split(x_bg,nT2/windowSz,1)
x_bg3 = np.asarray(x_bg2)
(dx1,dx2,dx3) = x_bg3.shape
x_bg4 = np.reshape(x_bg3,(dx1*dx2,dx3))
noBgSamples = x_bg4.shape
y_bg = 0*np.ones(noBgSamples[0])
noBgSamples = int(np.round(0.5 * noBgSamples[0]))
# normalize data
bgmin = np.amin(x_bg4, axis=1)
fgmin = np.amin(x_fg4, axis=1)
x_bg4 = ( x_bg4 - bgmin[:,None] )
x_fg4 = ( x_fg4 - fgmin[:,None] )
#bgmax = np.amax(x_bg4, axis=1)
#fgmax = np.amax(x_fg4, axis=1)
#x_fg4 = x_fg4 / fgmax[:, None]
#x_bg4 = x_bg4 / bgmax[:, None]

X_train = np.concatenate((x_fg4[0:noFgSamples , :], x_bg4[0:noBgSamples , :]), axis = 0)
Y_train = np.concatenate((y_fg[0:noFgSamples], y_bg[0:noBgSamples]), axis = 0)
X_test = np.concatenate((x_fg4[noFgSamples: , :], x_bg4[noBgSamples: , :]), axis = 0)
Y_test = np.concatenate((y_fg[noFgSamples:], y_bg[noBgSamples:]), axis = 0)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#############
### Predict Double Exp Parameters
#############
x_Train_Pred = model_heatingParamClassifier.predict(X_train)
X_Test_Pred = model_heatingParamClassifier.predict(X_test)

#############
### Train Classifier
#############
#TODO: maybe it'd be better to include the layers of our previously trained model ...
print('-> train classifier')
input_doubleExpModel = Input(shape=(doubleExpModelOutputDim,))
sm = Dense(10, activation='linear', name="classifier_intermediate")(input_doubleExpModel)
sm2 = Dense(1, activation='sigmoid', name="classifier")(sm)
model = Model(input = input_doubleExpModel, output = sm2)
model.compile(optimizer='adam', metrics={'classifier': 'accuracy'},
              loss={'classifier': 'binary_crossentropy'},
              )
model.fit(x_Train_Pred, Y_train, batch_size=32, nb_epoch=noEpochs,
          validation_data=(X_Test_Pred, Y_test))
score = model.evaluate(X_Test_Pred, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save_weights('model_heatingClassifier', overwrite=True)
