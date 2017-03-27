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
    def my_init(self,n,d,k):
        self.hidden_dim = 16
        self.step_size = 1
        self.reg = 0#1e-3
        h = self.hidden_dim  # size of hidden layer

        self.W1 = self.initialize_weights(d, h)
        self.Wh = self.initialize_weights(h, h)
        self.b1 = np.zeros((1, h))                  # h-bias, 1 for each hidden node/classifier

        self.W2 = self.initialize_weights(h, k)     # h-input dimensions, k-nodes (size/dmin of word2vec of output language)
        self.b2 = np.zeros((1, k))                  # k-bias, 1 for each output layer node


    #out of three initializations np.random.randn(d, k) is giving better/faster convergence than other techniques
    def initialize_weights(self,d,k):   #dimensions,classes
        return np.random.randn(d, k) #* np.sqrt(2.0/d)        #Calibrating the variances with 1/sqrt(n).
        # return 2*np.random.random((d,k)) - 1


# StochasticGradientDescent: learn the weights after each example. X contains one example
#     Input: 1 sentence(English/French) context vector,
#       @params:
#       X np 2dmin array
#           1st row: context vector for word 1 of Engligh sentence
#           2nd row: context vector for word 2 of same English sentence
#           ::
#       y np 2dmin array
#           1st row: context vector for word 1 of corresponding French sentence
#          2nd row: context vector for word 2 of same French sentence
#           ::
#     Output:
#       1. predicted context vector for each French word. List of 1D np-array
#       2. Error in prediction. Numeric
#       3. gradient on context vector (d(context_w1),d(context_w2)...). List of 1D np-array

    def train(self, X, y):
        n, d = X.shape          # there are n words(n timestamps) having d features each. (d = size of 1 word vector in given language)
        k = y.shape[1]          # k = size of 1 word vector in target language

        self.my_init(n,d,k)     #called just once

        dW2 = np.zeros_like(self.W2)
        dWh = np.zeros_like(self.Wh)
        dW1 = np.zeros_like(self.W1)
        db2 = np.zeros_like(self.b2)
        db1 = np.zeros_like(self.b1)

        h = np.zeros((n+1, 1, self.hidden_dim))  #h[t] stores output hidden layer at time t(i.e., (t-1)th example). [[1,2,3...16]]
        dh = np.zeros((n+1, 1, self.hidden_dim))
        total_error = 0
        output = []
        dX = []
        # Feed-forward--------------------------------
        for i in xrange(n):  # i= 0,1,2,..n-1    (w1 w2 w3)
            t=i+1
            Xi = np.array(X[i],ndmin=2)              # 1xd
            yi = np.array(y[i],ndmin=2)              # 1xk
            # print Xi,yi

            net1 = np.dot(Xi, self.W1) + np.dot(h[t-1], self.Wh) + self.b1       #h[0]=0. X[i]=1xd, W=dxh
            h[t] = 1/(1+np.exp(-net1))               # 1xh

            net2 = np.dot(h[t], self.W2) + self.b2   # 1xh, hxk > 1xk. Eg: [[1,2,3]]
            scores = 1/(1+np.exp(-net2))

            total_error += 0.5 * np.sum(np.square(yi - scores))
            output.append(scores[0])

            dscore = -(yi - scores)
            dnet2 = dscore * (scores * (1 - scores))

            dW2 += np.dot(h[t].T, dnet2)                    #accumulate     hx1,1x1 > hx1
            db2 += np.sum(dnet2, axis=0, keepdims=True)     #accumulate
            dh[t] = np.dot(dnet2, self.W2.T)                #cannot accumulate. Need exact value in future.


        # Backward--------------------------------
        dnet1_next_layer = np.zeros((1, self.hidden_dim))
        for t in xrange(n,0,-1):  # t = d,d-1,d-2,..1
            i=t-1
            Xi = np.array(X[i], ndmin=2)

            dht = dh[t] + np.dot(dnet1_next_layer, self.Wh.T)   #From_output_layer + h[t]_was_also_input_to_next_hidden_layer

            dnet1 = dht * (h[t] * (1-h[t]))                 # 1xh, 1xh

            dWh += np.dot(h[t - 1].T, dnet1)                # h[0] = 0
            dW1 += np.dot(Xi.T, dnet1)
            db1 += np.sum(dnet1, axis=0,keepdims=True)

            dX.append(np.dot(dnet1, self.W1.T)[0])          #1xh, (dxh).T > 1xd
            dnet1_next_layer = dnet1

        # regularization gradient
        dW2 += self.reg * self.W2
        dWh += self.reg * self.Wh
        dW1 += self.reg * self.W1

        self.W2 += (self.step_size * (-dW2))
        self.b2 += (self.step_size * (-db2))
        self.Wh += (self.step_size * (-dWh))
        self.W1 += (self.step_size * (-dW1))
        self.b1 += (self.step_size * (-db1))

        dX.reverse()    #reverse the order so it becomes d(w1),d(w2)...

        return output, total_error, dX

    def predict(self, X):
        n, d = X.shape  # 2x1
        h = np.zeros((n+1, self.hidden_dim))  #h[t] stores output hidden layer at time t(i.e., (t-1)th example).
        scores=[]

        # Feed-forward--------------------------------
        for i in xrange(n):  # i= 0,1,2,..n-1    (word0 word1 word2)
            t=i+1
            net1 = np.dot(X[i], self.W1) + np.dot(h[t-1], self.Wh) +self.b1       #h[0]=0. X[i]=1xd, W=dxh
            h[t] = 1/(1+np.exp(-net1))

            net2 = np.dot(h[t], self.W2)+self.b2
            scores.append(1/(1+np.exp(-net2)))

        return np.array([scores]).T

#End Class NeuralNetClassifier-----------------------------------
