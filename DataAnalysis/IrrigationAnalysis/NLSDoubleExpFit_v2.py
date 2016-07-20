import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

from theano import function, config, shared, sandbox
import theano.sandbox.cuda.basic_ops

from IrrigationDetector_DoubleExp_v2 import generateTrainingSample
from IrrigationDetector_DoubleExp_v2 import jacobian
from IrrigationDetector_DoubleExp_v2 import doubleExponentialFunction
from TrainHeatingClassifier import loadModelParamClassifier
from IrrigationDetector_DoubleExp_v2 import scaleParameters
from IrrigationDetector_DoubleExp_v2 import unscaleParameters
from IrrigationDetector_DoubleExp_v2 import scaleData
from scipy.optimize import minimize, least_squares 
### Constants
windowSz = 300
eps = 1e-5
noIter = 10000
alpha = 0.1
(y,param) = generateTrainingSample(1,300)
print("\n\nx*: " + str(param))
yraw = y

### LBFGS-B fit
def doubleExp_jacWrapper(params,y,t):
        x = np.array(params)
	(J,f) = jacobian(x,t,y)
	return J

def doubleExp_wrapper(params, y,t):
	x = np.array(params)
	yh = doubleExponentialFunction(t,x)
	f = y-yh
	return f

def lmFit(y,x,t):
	x = x[0,]
	y = y[0,]
	print("constrained NLS using trust-region Levenberg-Marquardt")
	#mybounds = ([-np.inf,-np.inf,-np.inf,-np.inf,20], [0,0,0,0,36])
	#scl = [1, 1, 1, 1, 36]
	#mybounds = [(None,None), (None,None), (None,None), (None,None), (20, None)]
	res = least_squares(doubleExp_wrapper, np.array(x),method='lm', args=(np.array(y), np.array(t)), jac=doubleExp_jacWrapper) #bounds=mybounds, x_scale=scl)
	print("converged at " + str(res.nfev) + " f: " + str(np.linalg.norm(res.fun)) + " => " + str(res.x))
	return (res.nfev,res.x,res.fun)

### Gauss-Newton fit
def gnFit(y,x,t):
	print("NLS using Gauss-Newton optimization")
	#print("\n\n*** Starting optimization scheme with near optimal initialization")
	#x = param + np.random.rand(1, 4)
	#t = np.linspace(0, 4, windowSz)
	#print("x_no: " + str(x))
	# Gauss-Newton iteration
	try:
		for i in range(0,10000):
			(J,f) = jacobian(x.transpose(),t,y)
			#print("f: " + str(np.linalg.norm(f)))
			Jt = J.transpose()
			JtJ = Jt.dot(J)
			mJtf = -1 * Jt.dot(f)
			#solve: JtJ * h = mJtf
			h = np.linalg.solve(JtJ,mJtf)
			xold = x
			x = x + alpha * h.transpose()
			# fix param orientation for comparison to optimal solution
			if(np.abs(x[0,0]) > np.abs(x[0,1])):
				tmp = x[0,0]
				x[0,0] = x[0,1]
				x[0,1] = tmp
				tmp = x[0,2]
				x[0,2] = x[0,3]
				x[0,3] = tmp

			if(np.linalg.norm(f) < eps):
				print("converged at " + str(i) + ": f: " + str(np.linalg.norm(f)) + " => x: " + str(x) + " eps: " + str(np.linalg.norm(x-param)))
				break
		if(i == 999):
			print("hit maxIter at " + str(i) + ": f: " + str(np.linalg.norm(f)) + " => x: " + str(x) + " eps: " + str(np.linalg.norm(x-param)))
	except np.linalg.linalg.LinAlgError as err:
		print("ERR => " + err.message)
		i=np.nan
		x=xold
		f=np.nan
	x = x[0,:]
	return (i,x,f)


def main(noSamples):

	data = 0*np.ones((noSamples, 16))

	for k in range(0, noSamples):
		(y,param) = generateTrainingSample(1,300)
		yraw = y
		print("\n\n*** Starting optimization scheme with near optimal initialization")
		x = param + 0.1*np.random.rand(1, 5)
		t = np.linspace(0, 4, windowSz)
		print("x_no: " + str(x))
		#(i,x,f) = gnFit(y,x,t)
		t1 = time.time()
		(i,x,f) = lmFit(y,x,t)
		t2 = time.time() - t1
		gn_no_noIterations = i
		gn_no_ss = np.linalg.norm(f)
		gn_no_epsParam = np.linalg.norm(x[0:1]-param[0,0:1])
		gn_no_eps = np.linalg.norm(x.transpose()-param[0,])
		data[k, 0] = gn_no_noIterations
		data[k, 1] = t2
		data[k, 2] = gn_no_epsParam
		data[k, 3] = gn_no_eps

		print("\n\n*** Starting optimization scheme with AI initialization")
		pModelWeights = "model_weights_v2"
		pModelConfig = "model_config_v2"
		heatingParamAI = loadModelParamClassifier(pModelConfig, pModelWeights)
		t = np.linspace(0, 4, windowSz)
		t1 = time.time()
		x = heatingParamAI.predict(scaleData(y))
		t2 = time.time() - t1
		x = unscaleParameters(x)
		y = y*34
		yh = doubleExponentialFunction(t,x[0,])
		f = y - yh
		ai_noIterations = 1
		ai_ss = np.linalg.norm(f)
		ai_epsParam = np.linalg.norm(x[0,0:1]-param[0,0:1])
		ai_eps = np.linalg.norm(x-param[0,])
		data[k, 4] = ai_noIterations
		data[k, 5] = t2
		data[k, 6] = ai_epsParam
		data[k, 7] = ai_eps
		print("x_ai: " + str(x))
		#(i,x,f) = gnFit(y,x,t)
		t1 = time.time()
		(i,x,f) = lmFit(y,x,t)
		t2 = time.time() - t1
		gn_ai_noIterations = i
		gn_ai_ss = np.linalg.norm(f)
		gn_ai_epsParam = np.linalg.norm(x[0:1]-param[0,0:1])
		gn_ai_eps = np.linalg.norm(x-param[0,])
		data[k, 8] = gn_ai_noIterations
		data[k, 9] = t2
		data[k, 10] = gn_ai_epsParam
		data[k, 11] = gn_ai_eps

		print("\n\n*** Starting optimization scheme with random initialization")
		x = -1 * np.random.rand(1, 5)
		x[0,4] = 2*x[0,4] + 28 
		t = np.linspace(0, 4, windowSz)
		print("x_rnd: " + str(x))
		#(i,x,f) = gnFit(y,x,t)
		t1 = time.time()
		(i,x,f) = lmFit(y,x,t)
		t2 = time.time() - t1
		gn_rand_noIterations = i
		gn_rand_ss = np.linalg.norm(f)
		gn_rand_epsParam = np.linalg.norm(x[0:1]-param[0,0:1])
		gn_rand_eps = np.linalg.norm(x-param[0,])
		data[k, 12] = gn_rand_noIterations
		data[k, 13] = t2
		data[k, 14] = gn_rand_epsParam
		data[k, 15] = gn_rand_eps


		print("\n\nStatistics")
		print("==========")
		print("NO : " + str(gn_no_noIterations) + " ss " + str(gn_no_ss) + " epsParam " + str(gn_no_epsParam) + " eps " + str(gn_no_eps))
		print("AI : " + str(ai_noIterations) + " ss " + str(ai_ss) + " epsParam " + str(ai_epsParam) + " eps " + str(ai_eps))
		print("AILM : " + str(gn_ai_noIterations) + " ss " + str(gn_ai_ss) + " epsParam " + str(gn_ai_epsParam) + " eps " + str(gn_ai_eps))
		print("RND: " + str(gn_rand_noIterations) + " ss " + str(gn_rand_ss) + " epsParam " + str(gn_rand_epsParam) + " eps " + str(gn_rand_eps))

	b = open('test.csv', 'w')
	a = csv.writer(b)
	a.writerows(data)
	b.close()

#print("converged.")
