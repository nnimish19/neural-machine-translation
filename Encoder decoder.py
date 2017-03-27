import numpy as np 
import pandas as pd
import word2vec as w2v
import decoder as decoder

def tanh(x):
    return np.tanh(x)

def tanh_d(output):
    return 1-(output**2)

class encoderRNN():
    def __init__(self,m,n):
        self.model = m
        self.nodes = n
        self.hidden_states = [0]*n
        self.create_weights()

    def create_weights(self):
        self.output_weights = np.random.rand(self.nodes)
        self.output_bias = np.random.rand(1)
        self.hidden_weights = np.random.rand(self.nodes,100)
        self.hidden_bias = np.random.rand(self.nodes)
        self.hidden_state_wts = np.random.rand(self.nodes)


    def RNN(self,x):
        # X.W product
        h  = np.dot(self.hidden_weights,x)

        #hidden state from the layer

        hidden_ouput = tanh(h + np.multiply(self.hidden_states,self.hidden_state_wts) + self.hidden_bias)

        #output net
        o_net = np.dot(self.output_weights,hidden_ouput)
        output = tanh(o_net+self.output_bias)
        self.hidden_state_wts = hidden_ouput
        return output[0]


    def train(self,data, Decoder):
        context_vector = []
        for i in data.split(" "):
            word_vec = self.model[i]
            context_vector.append(self.RNN(word_vec))
        # context_grad = Decoder.train()
        print(context_vector)


if __name__ == "__main__":
    eng_model = w2v.load('datasets/english.bin')
    fr_model = w2v.load('datasets/french.bin')
    eng_training_sentence = 'Resumption of the session\n'.rstrip()
    Encoder = encoderRNN(eng_model,10)
    Decoder = decoder.RNNClassifier()
    Encoder.train(eng_training_sentence,Decoder)


    