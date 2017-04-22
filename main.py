import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec as w2v
import encoder as encoder
import decoder as decoder

Encoder = encoder.RNNClassifier()
Decoder = decoder.RNNClassifier()

def plot_graph(n, loss_axis):
    # print loss_axis
    plt.xlabel("# ith Example", fontsize=15)
    plt.ylabel("Training Error", fontsize=15)
    plt.plot(np.arange(n), loss_axis, '-')
    plt.show()

def sentenceToVector(model, s):     #Array of words
    list= [model.wv[w] for w in s] #list of 2D np-array
    return np.array(list)

def vectorToSentence(model, V):    #Array of word_embeddings
    s=""
    for v in V:
        sim = model.wv.most_similar(positive=[v], topn=1)
        s+=sim[0][0]+" "
    return s

def trainScript(en_sentences,fr_sentences, en_model,fr_model):
    loopcount = len(en_sentences)
    loss_axis = []
    for i in xrange(loopcount):
        en_arr = en_sentences[i]                #["line","one"]
        fr_arr = fr_sentences[i]
        X = sentenceToVector(en_model, en_arr)  #[ [1,2,3], [4,7,1] ]
        Y = sentenceToVector(fr_model, fr_arr)
        # make X and Y to have same #words
        if X.shape[0]<Y.shape[0]:
            Y = Y[:X.shape[0]]
        else:
            X = X[:Y.shape[0]]

        # print "count: ",i
        pred_vector, error = Encoder.train(X, Decoder, Y)
        loss_axis.append(error)
        # print vectorToSentence(fr_model, pred_vector), "\n----------------\n"

    plot_graph(loopcount, loss_axis)

if __name__ == "__main__":
    f= open("datasets/en_sample.txt","r")
    text = f.read()
    lines = text.split("\n")
    en_sentences = [l.split(" ") for l in lines]
    f.close()

    f = open("datasets/fr_sample.txt", "r")
    text = f.read()
    lines = text.split("\n")
    fr_sentences = [l.split(" ") for l in lines]
    f.close()

    # model = w2v([["line","one"], ["This", "is", "line", "two"]], size=2, window=2, min_count=1, workers=1)
    en_model = w2v(en_sentences, size=5, window=5, min_count=1, workers=4)
    fr_model = w2v(fr_sentences, size=5, window=5, min_count=1, workers=4)

    # model.save(fname)
    # en_model = w2v.load('datasets/english.bin')
    # fr_model = w2v.load('datasets/french.bin')

    # trainScript([en_sentences[0]], [fr_sentences[0]], en_model, fr_model)
    trainScript(en_sentences, fr_sentences, en_model, fr_model)