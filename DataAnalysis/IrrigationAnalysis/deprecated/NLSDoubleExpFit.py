import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from IrrigationDetector_DoubleExp import generateTrainingSample
from IrrigationDetector_DoubleExp import jacobian
from IrrigationDetector_DoubleExp import doubleExponentialFunction
from TrainHeatingClassifier import loadModelParamClassifier

### Constants
windowSz = 300
eps = 1e-5
noIter = 10000
alpha = 0.01
(y,param) = generateTrainingSample(1,300)
print("\n\nx*: " + str(param))

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
			x = x + alpha * h.transpose()
			# fix param orientation for comparison to optimal solution
			if(x[0,0] > x[0,1]):
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
		x=np.nan
		f=np.nan
	return (i,x,f)




print("\n\n*** Starting optimization scheme with near optimal initialization")
x = param + np.random.rand(1, 4)
t = np.linspace(0, 4, windowSz)
print("x_no: " + str(x))
(i,x,f) = gnFit(y,x,t)
gn_no_noIterations = i
gn_no_ss = np.linalg.norm(f)
gn_no_epsParam = np.linalg.norm(x-param)



print("\n\n*** Starting optimization scheme with AI initialization")
pModelWeights = "model_weights"
pModelConfig = "model_config"
heatingParamAI = loadModelParamClassifier(pModelConfig, pModelWeights)
t = np.linspace(0, 4, windowSz)
x = heatingParamAI.predict(y)
print("x_ai: " + str(x))
(i,x,f) = gnFit(y,x,t)
gn_ai_noIterations = i
gn_ai_ss = np.linalg.norm(f)
gn_ai_epsParam = np.linalg.norm(x-param)



print("\n\n*** Starting optimization scheme with random initialization")
x = np.random.rand(1, 4)
t = np.linspace(0, 4, windowSz)
print("x_rnd: " + str(x))
(i,x,f) = gnFit(y,x,t)
gn_rand_noIterations = i
gn_rand_ss = np.linalg.norm(f)
gn_rand_epsParam = np.linalg.norm(x-param)



print("\n\nStatistics")
print("==========")
print("NO : " + str(gn_no_noIterations) + " ss " + str(gn_no_ss) + " epsParam " + str(gn_no_epsParam))
print("AI : " + str(gn_ai_noIterations) + " ss " + str(gn_ai_ss) + " epsParam " + str(gn_ai_epsParam))
print("RND: " + str(gn_rand_noIterations) + " ss " + str(gn_rand_ss) + " epsParam " + str(gn_rand_epsParam))



#print("converged.")
