import json
import h5py
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import SReLU
from keras.layers import Input
from keras.layers.recurrent import LSTM

from sklearn import tree
from sklearn.cross_validation import cross_val_score

from IrrigationDetector_DoubleExp_v2 import scaleParameters
from IrrigationDetector_DoubleExp_v2 import unscaleParameters
from IrrigationDetector_DoubleExp_v2 import scaleData

def loadModelParamClassifier(pModelConfig, pModelWeights):
	#print('-> importing previously trained model')
	f = open(pModelConfig,"r")
	model_def = json.load(f)
	model_heatingParamClassifier = model_from_json(model_def)
	model_heatingParamClassifier.load_weights(pModelWeights)
	model_heatingParamClassifier.compile(loss='mse', optimizer='adam')
	#model_heatingParamClassifier.summary()
	return model_heatingParamClassifier

def main():
	#############
	### CONSTANTS
	#############
	windowSz = 300
	hiddenSz = 90
	noEpochs = 200
	batchSz = 65536
	pData = "irr_dr_training.mat"
	pModelWeights = "model_weights_v2"
	pModelConfig = "model_config_v2"

	#############
	### Load Model for classifying double exponential model parameters
	#############
	print('-> importing previously trained model')
	model_heatingParamClassifier = loadModelParamClassifier(pModelConfig, pModelWeights)

	#############
	### READ DATA
	#############
	print('-> reading data')
	f = h5py.File(pData,"r")
	bg = f["seqBackground"]
	fg = f["seqSpuelung"]
	x_c1 = scaleData(np.array(fg.value))
	x_c2 = scaleData(np.array(bg.value))
	f.close()
	x_fg = np.transpose(x_c1[601:901,])
	x_bg = np.transpose(x_c2[601:901,])

	#x_fg = np.transpose(x_c1[601:901,0:9402])
	#x_bg = np.transpose(x_c2[601:901,0:9402])

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
	noFgSamples = int(np.round(0.8 * noFgSamples[0] ))
	# background samples
	(nX2,nT2) = x_bg.shape
	x_bg2 = np.split(x_bg,nT2/windowSz,1)
	x_bg3 = np.asarray(x_bg2)
	(dx1,dx2,dx3) = x_bg3.shape
	x_bg4 = np.reshape(x_bg3,(dx1*dx2,dx3))
	noBgSamples = x_bg4.shape
	y_bg = 0*np.ones(noBgSamples[0])
	noBgSamples = int(np.round(0.8 * noBgSamples[0]))
	# normalize data
	#bgmin = np.amin(x_bg4, axis=1)
	#fgmin = np.amin(x_fg4, axis=1)
	#x_bg4 = ( x_bg4 - bgmin[:,None] )
	#x_fg4 = ( x_fg4 - fgmin[:,None] )
	
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
	X_Train_Pred = model_heatingParamClassifier.predict((X_train))
	X_Test_Pred = model_heatingParamClassifier.predict((X_test))
	X_Train_Pred = X_Train_Pred[:,0:2]
	#Y_train = Y_train[0:9401]
	#Y_test = Y_test[0:9401]
	X_Test_Pred = X_Test_Pred[:,0:2]
	szData = X_Test_Pred.shape
	doubleExpModelOutputDim = szData[1]
	print("input sz: " + str(doubleExpModelOutputDim))
	#############
	### Train DT Classifier
	#############
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_Train_Pred, Y_train)
	score_pred = clf.score(X_Train_Pred, Y_train)
	score_test = clf.score(X_Test_Pred, Y_test)
	print(str(score_pred) + " vs " + str(score_test))
	cvres = cross_val_score(clf, X_Test_Pred, Y_test, cv=10)
	print(str(cvres))
	#############
	### Train NN Classifier
	#############
	print('-> train NN classifier')
	input_doubleExpModel = Input(shape=(doubleExpModelOutputDim,))
	
	model = Sequential()
	model.add(Dense(5, input_dim=doubleExpModelOutputDim, activation='linear'))
	model.add(SReLU())
	model.add(Dropout(0.05))
	model.add(Dense(50, activation='sigmoid'))
#	model.add(Dropout(0.05))
	model.add(Dense(1, activation='sigmoid'))
#	sm = Dense(5, activation='linear', name="classifier_intermediate")(input_doubleExpModel)
#	sm1 = SReLU()(sm)
#	sm11 = Dropout(0.05)(sm1)
#	sm12 = Dense(50, activation='linear')(sm11)
#	sm13 = Dropout(0.05)(sm12)	
#	sm2 = Dense(1, activation='sigmoid', name="classifier")(sm13)
#	model = Model(input = input_doubleExpModel, output = sm2)
	model.compile(optimizer='rmsprop',  metrics={'dense_3': 'accuracy'},
	              loss={'dense_3': 'binary_crossentropy'}
	              )
	model.fit(X_Train_Pred, Y_train, batch_size=batchSz, nb_epoch=noEpochs,  
	          validation_data=(X_Test_Pred, Y_test))
	score = model.evaluate(X_Test_Pred, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	model.save_weights('heatingClassifier_weights', overwrite=True)
    	json_rep = model.to_json()
    	f = open('heatingClassifier_architecture',"wb")
    	json.dump(json_rep,f)
