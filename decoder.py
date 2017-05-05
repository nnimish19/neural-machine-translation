import numpy as np

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper

class RNNClassifier(object):
    def __init__(self):
        pass

    @run_once
    def my_init(self,d,k):
        self.hidden_dim = 64
        self.output_dim = k
        self.step_size = 1
        self.reg = 0#1e-3
        h = self.hidden_dim  # size of hidden layer

        self.W1 = self.initialize_weights(d, h)
        self.Wh = self.initialize_weights(h, h)
        self.Wy = self.initialize_weights(k, h)
        self.b1 = np.zeros((1, h))                  # h-bias, 1 for each hidden node/classifier

        self.W2 = self.initialize_weights(h, k)     # h-input dimensions, k-nodes (size/dmin of word2vec of output language)
        self.b2 = np.zeros((1, k))                  # k-bias, 1 for each output layer node


    #out of three initializations np.random.randn(d, k) is giving better/faster convergence than other techniques
    def initialize_weights(self,d,k):   #dimensions,classes
        return np.random.randn(d, k) #* np.sqrt(2.0/d)        #Calibrating the variances with 1/sqrt(n).
        # return 2*np.random.random((d,k)) - 1


# StochasticGradientDescent: learn the weights after each example. X contains one example
#     Input: 1 sentence(English) context vector,
#       @params:
#       C: np-2D array: [1xh]   where h = #hidden_layers in encoder.
#                              As an input to decoder its just #dimension=d of input vector

#       Y: np 2dmin array
#           1st row: Actual word vector for word 1 of corresponding French sentence
#           2nd row: Actual vector for word 2 of same French sentence
#           ::
#     Output:
#       1. predicted sentence_vector in French. List of 1D np-array (word_vector for w1, w2 ...)
#       2. Error in prediction. Numeric
#       3. gradient on context vector (d(context_w1),d(context_w2)...). List of 1D np-array

    def train(self, C, Y):

        n, k = Y.shape  # k = size of 1 word vector in target language. Note X = nxd, Y = nxk
        d = C.shape[1]  # d = size of 1 context vector

        X = np.repeat(C, n, axis=0)         #for convinience: one context vector for every output word

        # Now X has there are n context words(n timestamps) having d features each.


        self.my_init(d,k)     #called just once

        dW2 = np.zeros_like(self.W2)
        dWh = np.zeros_like(self.Wh)
        dWy = np.zeros_like(self.Wy)
        dW1 = np.zeros_like(self.W1)
        db2 = np.zeros_like(self.b2)
        db1 = np.zeros_like(self.b1)

        h = np.zeros((n+1, 1, self.hidden_dim))  #h[t] stores output hidden layer at time t(i.e., (t-1)th example). [[1,2,3...16]]
        output = np.zeros((n + 1, k))            #stores outcome(predicted word vector) of output layer. n= number of words in Target Language

        total_error = 0
        dX = np.zeros_like(X)

        # Feed-forward--------------------------------
        for i in xrange(n):  # i= 0,1,2,..n-1    (w1 w2 w3)
            t=i+1
            Xi = np.array(X[i],ndmin=2)              # 1xd
            yi = np.array(Y[i],ndmin=2)              # 1xk
            # print Xi,yi

            net1 = np.dot(Xi, self.W1) + np.dot(h[t-1], self.Wh) + np.dot(output[t-1], self.Wy) + self.b1       #h[0]=0. X[i]=1xd, W1=dxh,
            h[t] = 1/(1+np.exp(-net1))               # 1xh

            net2 = np.dot(h[t], self.W2) + self.b2   # 1xh, hxk > 1xk. Eg: [[1,2,3]]
            scores = 1/(1+np.exp(-net2))
            output[t-1] = scores[0]

            total_error += 0.5 * np.sum(np.square(yi - scores))

        scores = output[1:]

        # Backward--------------------------------
        dnet1_next_layer = np.zeros((1, self.hidden_dim))
        for t in xrange(n,0,-1):  # t = d,d-1,d-2,..1
            i=t-1
            Xi = np.array(X[i], ndmin=2)
            yi = np.array(Y[i], ndmin=2)

            dscore = -(yi - scores[i])                      # 1xk,1xk

            dnet2 = dscore * (scores[i] * (1 - scores[i]))  # 1xk
            dW2 += np.dot(h[t].T, dnet2)                    # accumulate     hx1,1xk > hxk
            db2 += np.sum(dnet2, axis=0, keepdims=True)     # accumulate
            dht = np.dot(dnet2, self.W2.T) + np.dot(dnet1_next_layer, self.Wh.T)  # From_output_layer + h[t]_was_also_input_to_next_hidden_layer
            #              1xk      (hxk).T                 1xh           hxh

            dnet1 = dht * (h[t] * (1-h[t]))                 # 1xh, 1xh
            dWh += np.dot(h[t - 1].T, dnet1)                # h[0] = 0
            dWy += np.dot(np.array(output[t - 1], ndmin=2).T, dnet1)           # output[0] = 0
            dW1 += np.dot(Xi.T, dnet1)
            db1 += np.sum(dnet1, axis=0,keepdims=True)

            dX[i] = np.dot(dnet1, self.W1.T)[0]          #1xh, (dxh).T > 1xd
            dnet1_next_layer = dnet1

        # regularization gradient
        dW2 += self.reg * self.W2
        dWh += self.reg * self.Wh
        dWy += self.reg * self.Wy
        dW1 += self.reg * self.W1

        self.W2 += (self.step_size * (-dW2))
        self.b2 += (self.step_size * (-db2))
        self.Wh += (self.step_size * (-dWh))
        self.Wy += (self.step_size * (-dWy))
        self.W1 += (self.step_size * (-dW1))
        self.b1 += (self.step_size * (-db1))

        dX = np.sum(dX, axis=0)  # sum across gradients. 1xh

        return output[1:], total_error, dX

    def get_weights(self):
        return (self.W2,self.b2,self.Wh,self.Wy,self.W1,self.b1)

    def set_weights(self,Decoder_weights):
        self.W2, self.b2, self.Wh, self.Wy, self.W1, self.b1 = Decoder_weights['W2'],Decoder_weights['b2'],Decoder_weights['Wh'],Decoder_weights['Wy'],Decoder_weights['W1'],Decoder_weights['b1']
        self.hidden_dim,self.output_dim = self.W2.shape


    def predict(self, C, n):                     # Context Vector 1xd, #words in english sentence

        X = np.repeat(C, n, axis=0)              # for convinience: one context vector for every output word

        h = np.zeros((n + 1, 1, self.hidden_dim))       #h[t] stores output hidden layer at time t(i.e., (t-1)th example). [[1,2,3...16]]
        output = np.zeros((n + 1, self.output_dim))     #stores outcome(predicted word vector) of output layer. n= number of words in Target Language

        # Feed-forward--------------------------------
        for i in xrange(n):  # i= 0,1,2,..n-1    (w1 w2 w3)
            t=i+1
            Xi = np.array(X[i],ndmin=2)                # 1xd
            # yi = np.array(Y[i],ndmin=2)              # 1xk
            # print Xi,yi

            net1 = np.dot(Xi, self.W1) + np.dot(h[t-1], self.Wh) + np.dot(output[t-1], self.Wy) + self.b1       #h[0]=0. X[i]=1xd, W1=dxh,
            h[t] = 1/(1+np.exp(-net1))               # 1xh

            net2 = np.dot(h[t], self.W2) + self.b2   # 1xh, hxk > 1xk. Eg: [[1,2,3]]
            scores = 1/(1+np.exp(-net2))
            output[t-1] = scores[0]

        return output[1:]

#End Class RNNClassifier-----------------------------------
