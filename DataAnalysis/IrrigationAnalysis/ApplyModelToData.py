import h5py
import time
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from IrrigationDetector_DoubleExp_v2 import generateTrainingSample
from IrrigationDetector_DoubleExp_v2 import jacobian
from IrrigationDetector_DoubleExp_v2 import doubleExponentialFunction
from TrainHeatingClassifier import loadModelParamClassifier
from IrrigationDetector_DoubleExp_v2 import scaleParameters
from IrrigationDetector_DoubleExp_v2 import unscaleParameters
from IrrigationDetector_DoubleExp_v2 import scaleData
from scipy.optimize import minimize, least_squares 


def main(pData, windowSz):
	print("Loading data from hdf5 file " + str(pData))
        f = h5py.File(pData,"r")
        data = f["S"]
	print("Sz: " + str(data.shape))
	y = np.array(data.value).transpose()
	y = y[:,0:300]
	print("\n\n*** Deep Parameter Prediction")
	pModelWeights = "model_weights_v2"
	pModelConfig = "model_config_v2"
	pClassifierWeights = "heatingClassifier_weights"
	pClassifierConfig = "heatingClassifier_architecture"
	heatingParamAI = loadModelParamClassifier(pModelConfig, pModelWeights)
	t1 = time.time()
	x = heatingParamAI.predict(scaleData(y))
	t2 = time.time() - t1
	print("dt: " + str(t2))
	x_ai = unscaleParameters(x)
	print("x_ai: " + str(x))
	heatingClassifier = loadModelParamClassifier(pClassifierConfig, pClassifierWeights)
	x_pred = x[:,0:2]
	y_class = heatingClassifier.predict_classes(x_pred)
	#(i,x,f) = gnFit(y,x,t)
	#(i,x,f) = lmFit(y,x,t)
	#gn_ai_noIterations = i
	#gn_ai_ss = np.linalg.norm(f)
	#gn_ai_epsParam = np.linalg.norm(x[0:1]-param[0,0:1])
	print(str(y_class))
	##################
	### store data
	##################
	with h5py.File('results.h5', 'w') as hf:
		hf.create_dataset('x_ai', data=x_ai)
		hf.create_dataset('y',data=y_class)
