 import numpy as np

class two_hl_net(object):
    def __init__(self, n=3, lr=0.1, act='sigmod'):
        # initial neural networks layers
        self.hidden_weight_1 =  np.random.uniform(size=(2,n))
        self.hidden_weight_2 =  np.random.uniform(size=(n,n))
        self.output_weight =  np.random.uniform(size=(n,1))
        # initial output for each layer
        self.hidden_output_1 = None
        self.hidden_output_2 = None
        self.output = None
        # training parameters
        self.mode = 'train'
        self.lr = lr
        self.act = act
    
    def activation(self, X):
        if self.act == 'sigmod':
            return 1.0/(1.0 + np.exp(-X))
        if self.act == 'tanh':
            return np.tanh(X)
        if self.act == 'no':
            return X
    
    def derivative_activation(self, X):
        if self.act == 'sigmod':
            return X * (1.0 - X)
        if self.act == 'tanh':
            return 1.0 - X**2
        if self.act == 'no':
            return X

    def forward(self, X):
        # forward propagation
        self.hidden_output_1 = self.activation(np.dot(X, self.hidden_weight_1))
        self.hidden_output_2 = self.activation(np.dot(self.hidden_output_1, self.hidden_weight_2))
        self.output = self.activation(np.dot(self.hidden_output_2, self.output_weight))

        return self.output

    def backward(self,x,y):
        error_output = y - self.output
        if self.mode == 'train':
            # get loss and back propagation
            d_output = error_output*self.derivative_activation(self.output)
            
            error_h2 = d_output.dot(self.output_weight.T) 
            d_hidden_output_2 = error_h2*self.derivative_activation(self.hidden_output_2)

            error_h1 = error_h2.dot(self.hidden_weight_2.T)
            d_hidden_output_1 = error_h1*self.derivative_activation(self.hidden_output_1)
            
            # update weight
            self.output_weight += self.hidden_output_2.T.dot(d_output)*self.lr
            self.hidden_weight_2 += self.hidden_output_1.T.dot(d_hidden_output_2)*self.lr
            self.hidden_weight_1 += x.T.dot(d_hidden_output_1)*self.lr

        return abs(np.mean(error_output))