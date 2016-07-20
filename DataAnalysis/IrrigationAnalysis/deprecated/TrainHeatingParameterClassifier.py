import h5py
import numpy as np
#get_ipython().magic(u'matplotlib notebook')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.regularizers import l1,l2,l1l2, activity_l2

#############
### CONSTANTS
#############
rndSeed = 23
noEpochs = 10
noTrainingSamples = 100000
noTestSamples = 10000
windowSz = 300
regulPen = l2(0.01)

###################
### Train regressor
###################

### TODO: WATCHOUT FOR OVERFITTING! -> L1/L2 penalty, DROPOUT layer, (SReLU)!

np.random.seed(rndSeed)
(x_train, y_train) = generateTrainingSample(noTrainingSamples, windowSz)
(x_test, y_test) = generateTrainingSample(noTestSamples, windowSz)
model = Sequential()
model.add(Dense(50, input_dim=windowSz, activation='sigmoid', init='he_uniform')) #, W_regularizer=regulPen))
#model.add(Dense(25, input_dim=100, activation='tanh'))
model.add(Dense(4, input_dim=50, activation='sigmoid'))#, W_regularizer=regulPen))
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,batch_size=32,nb_epoch=noEpochs)
score = model.evaluate(x_test, y_test, batch_size=32)
print("\n\nscore " + str(score))

model.save_weights('model_heatingParamClassifier', overwrite=True)
