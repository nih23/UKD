import h5py
import json
import numpy as np
#get_ipython().magic(u'matplotlib notebook')
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import SReLU
from keras.layers.recurrent import LSTM
from keras.regularizers import l1,l2,l1l2, activity_l2

def sampleAndStoreData(noTrainingSamples, noTestSamples):
    #############
    ### CONSTANTS
    #############
    rndSeed = 23
    windowSz = 300

    ###################
    ### sample data 
    ###################
    np.random.seed(rndSeed)
    (x_train, y_train) = generateTrainingSample(noTrainingSamples, windowSz)
    (x_test, y_test) = generateTrainingSample(noTestSamples, windowSz)

    ##################
    ### store data
    ##################
    with h5py.File('doubleExpData.h5', 'w') as hf:
       hf.create_dataset('x_train', data=x_train)
       hf.create_dataset('x_test', data=x_test)
       hf.create_dataset('y_train', data=y_train)
       hf.create_dataset('y_test', data=y_test)

def jacobian(x,ti,yi):
    J = np.array([-1 * x[2] * ti * np.exp(x[0]*ti), -1 * x[3] * ti * np.exp(x[1]*ti), -1 * np.exp(x[0]*ti), -1 * np.exp(x[1]*ti) ])
    yih = doubleExponentialFunction(ti, x)
    #yih = yih - yih.min()
    f = yi - yih
    return (J.transpose(),f.transpose())


def doubleExponentialFunction(ti, x):
    return x[2] * np.exp(x[0]*ti) + x[3] * np.exp(x[1]*ti)


def generateTrainingSample(noSamples, windowSz):
    x_train = np.ndarray(shape=(noSamples, windowSz), dtype=float, order='F')
    y_train = np.ndarray(shape=(noSamples, 4), dtype=float, order='F')
    
    for i in range(0,noSamples):
        l1 = 0.5*np.asscalar(np.random.rand(1,1))
        l2 = 0.5*np.asscalar(np.random.rand(1,1)) + 0.5
        dt1 = 6*np.asscalar(np.random.rand(1,1))
        dt2 = 6*np.asscalar(np.random.rand(1,1))
        #y_train[i,:] = np.array([dt1, dt2, l1, l2])
        y =  np.array([l1, l2, dt1, dt2])
        y_train[i,:] = y
        x = np.linspace(0, 4, windowSz)
        
        y = doubleExponentialFunction(x,y)
        #y = y - y.min()
        x_train[i,:] = y
        #plt.plot(x,y)
        #plt.show()
    
    return (x_train, y_train)

def main():
    #############
    ### CONSTANTS
    #############
    rndSeed = 23
    noEpochs = 100
    batchSz = 65536 
    windowSz = 300
    regulPen = l2(0.01)
    pModelConfig = "model_config"
    pWeights = "model_weights"

    ###################
    ### Train regressor
    ###################
    print("loading training data")
    with h5py.File('doubleExpData.h5','r') as hf:
       x_train = np.array(hf.get('x_train'))
       x_test = np.array(hf.get('x_test'))
       y_train = np.array(hf.get('y_train'))
       y_test = np.array(hf.get('y_test'))    
    np.random.seed(rndSeed)
    model = Sequential()
    model.add(Dense(50, input_dim=windowSz, activation='linear', W_regularizer=regulPen)) # , init='he_uniform'   % 5 # sigmoid
    model.add(SReLU())
    model.add(Dropout(0.05))
    model.add(Dense(50, activation='linear')) # tanh
    model.add(SReLU())
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='tanh'))
    model.add(Dense(4, activation='linear'))#, W_regularizer=regulPen)) % 4
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train,y_train,batch_size=batchSz,nb_epoch=noEpochs)
    score = model.evaluate(x_test, y_test, batch_size=batchSz)
    print("\n\nscore " + str(score))

    ### store model
    model.save_weights(pWeights, overwrite=True)
    json_rep = model.to_json()
    f = open(pModelConfig,"wb")
    json.dump(json_rep,f)
