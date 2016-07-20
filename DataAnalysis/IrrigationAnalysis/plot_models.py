from IPython.display import SVG
from keras.utils.visualize_util import plot, model_to_dot
from TrainHeatingClassifier import loadModelParamClassifier
import io

pModelWeights = "model_weights_v2"
pModelConfig = "model_config_v2"

heatingParamAI = loadModelParamClassifier(pModelConfig, pModelWeights)
svg_img = (model_to_dot(heatingParamAI, show_shapes=True).create(prog='dot', format='svg'))
file('heatingParamAI.svg', 'w').write(svg_img)
#plot(heatingParamAI, to_file='heatingParamAI.png', show_shapes=True)

