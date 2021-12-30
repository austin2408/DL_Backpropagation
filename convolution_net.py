import numpy as np
from generation_func import *


class Conv(object):
    def __init__(self, M=1, n=3, lr=0.01, act='sigmod'):
        # initial neural networks layers
        self.conv =  np.random.uniform(size=(2,M))
        self.hidden_weight_1 =  np.random.uniform(size=(M,n))
        self.output_weight =  np.random.uniform(size=(n,1))
        # initial output for each layer
        self.conv_output = None
        self.hidden_output_1 = None
        self.output = None
        # training parameters
        self.M = M
        self.mode = 'train'
        self.lr = lr
        self.act = act
    
    def activation(self, X):
        if self.act == 'sigmod':
            return 1.0/(1.0 + np.exp(-X))
        else:
            return np.tanh(X)
    
    def derivative_activation(self, X):
        if self.act == 'sigmod':
            return X * (1.0 - X)
        else:
            return 1.0 - X**2

    def forward(self, X):
        # forward propagation
        self.conv_output = np.zeros((X.shape[0], self.M))
        # do convolution
        for i in range(X.shape[0]):
            for j in range(self.M):
                self.conv_output[i,j] = np.sum(X[i]*self.conv[:,j])

        self.hidden_output_1 = self.activation(np.dot(self.conv_output, self.hidden_weight_1))
        self.output = self.activation(np.dot(self.hidden_output_1, self.output_weight))

        return self.output

    def backward(self, x, y):
        error_output = y - self.output
        if self.mode == 'train':
            d_output = error_output*self.derivative_activation(self.output)
            
            error_h1 = d_output.dot(self.output_weight.T) 
            d_hidden_output_1 = error_h1*self.derivative_activation(self.hidden_output_1)

            error_conv = d_hidden_output_1.dot(self.hidden_weight_1.T)
            d_conv_out = np.zeros((x.shape[0],self.M))
            for i in range(x.shape[0]):
                d_conv_out[i,:] = np.sum(x[i])
            d_conv = error_conv*d_conv_out

            # update weight
            self.output_weight += self.hidden_output_1.T.dot(d_output)*self.lr
            self.hidden_weight_1 += self.conv_output.T.dot(d_hidden_output_1)*self.lr
            self.conv += x.T.dot(d_conv)*self.lr

        return abs(np.mean(error_output))