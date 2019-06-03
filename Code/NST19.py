
# Libraries
from __future__ import print_function
import numpy as np
from keras.applications.vgg19 import VGG19
from PIL import Image
from keras import backend as K
from keras.models import Model
from scipy.optimize import fmin_l_bfgs_b
import time

# Inputs
height = 512
width = 512

# Content Image
CImPath = 'Examples/dancing.jpg'
CIm = Image.open(CImPath)
CIm = CIm.resize((width, height))
# CIm

# Style Image
SImPath = 'Examples/picasso.jpg'
SIm = Image.open(SImPath)
SIm = SIm.resize((width, height))

# Preprocessing
CArr = np.asarray(CIm, dtype='float32')
CArr = np.expand_dims(CArr, axis=0)
print(CArr.shape)

SArr = np.asarray(SIm, dtype='float32')
SArr = np.expand_dims(SArr, axis=0)
print(SArr.shape)

# Mean values of Imagenet on VGG16
CArr[:,:,:,0] -= 103.939
CArr[:,:,:,1] -= 116.779
CArr[:,:,:,2] -= 123.68
# RBG to BGR
CArr = CArr[:,:,:,::-1]

SArr[:,:,:,0] -= 103.939
SArr[:,:,:,1] -= 116.779
SArr[:,:,:,2] -= 123.68
# RBG to BGR
SArr = SArr[:,:,:,::-1]

# Variables
CIm = K.variable(CArr)
print(CIm.shape)
SIm = K.variable(SArr)
GIm = K.placeholder((1,height,width,3))
loss = K.variable(0.)

# Input Tensor
input_tensor = K.concatenate([CIm, SIm, GIm],axis=0)

# Model
model = VGG19(input_tensor=input_tensor, weights='imagenet',
              include_top=False)

# Layers
layers = dict([(layer.name, layer.output) for layer in model.layers])
layers

# Parameters
alpha = 0.025 # Content weight
beta = 5.0 # Style weight
gamma = 1.0 # Generated Image Weight

# Loss Functions
def ContentLoss(C, G):
    return K.sum(K.square(G-C))

layer_features = layers['block4_conv2']
CImFeatures = layer_features[0,:,:,:]
GImFeatures = layer_features[2,:,:,:]

loss += alpha*ContentLoss(CImFeatures, GImFeatures)

def gram_matrix(G):
    features = K.batch_flatten(K.permute_dimensions(G,(2,0,1)))
    G = K.dot(features, K.transpose(features))
    return G

def StyleLoss(S, G):
    S = gram_matrix(S)
    G = gram_matrix(G)
    Nl = 3
    Ml = width*height
    return K.sum(K.square(S-G))/(4. * (Nl**2) * (Ml**2) )

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

for name in feature_layers:
    layer_features = layers[name]
    style_features = layer_features[1,:,:,:]
    gen_features = layer_features[2,:,:,:]
    sl =StyleLoss(style_features,gen_features)
    loss += (sl/len(feature_layers))*beta

def VariationLoss(x):
    a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

loss += gamma * VariationLoss(GIm)

# Optimizing
grads = K.gradients(loss, GIm)

outputs = [loss]
outputs += grads
f_outputs = K.function([GIm], outputs) # initiates a function

def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# Output
x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

iterations = 100

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

x = x.reshape((height,width,3))
x = x[:,:,::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')

Image.fromarray(x)
