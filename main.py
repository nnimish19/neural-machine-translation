import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec as w2v
import encoder as encoder
import decoder as decoder
import timeit
import re
# from sklearn.metrics.pairwise import cosine_similarity as cs

Encoder = encoder.RNNClassifier()
Decoder = decoder.RNNClassifier()

def plot_graph(n, loss_axis):
    # print loss_axis
    plt.xlabel("# ith Example", fontsize=15)
    plt.ylabel("Training Error", fontsize=15)
    plt.plot(np.arange(n), loss_axis, '-')
    plt.show()

def saveWeights():
    Encoder_weights = {'W2': [], 'b2':[],'Wh':[],'W1':[], 'b1':[]}
    Decoder_weights = {'W2': [], 'b2': [], 'Wh': [], 'W1': [], 'b1': [],'Wy':[]}
    Encoder_weights['W2'], Encoder_weights['b2'], Encoder_weights['Wh'], Encoder_weights['W1'], Encoder_weights['b1'] = Encoder.get_weights()
    Decoder_weights['W2'], Decoder_weights['b2'], Decoder_weights['Wh'], Decoder_weights['Wy'], Decoder_weights['W1'], Decoder_weights['b1'] = Decoder.get_weights()
    np.save('save/encoder_weights.npy', Encoder_weights)
    np.save('save/decoder_weights.npy', Decoder_weights)

def sentenceToVector(model, s):     #s = Array of words
    list = []
    for w in s:
        if w in model.wv:
            list.append(model.wv[w])

    return np.array(list)

def vectorToSentence(model, V):    #V = Array of word_vectors
    s= ""
    for v in V:
        sim = model.wv.most_similar(positive=[v], topn=1)
        s+=sim[0][0]+" "
    return s

def trainScript(en_sentences, fr_sentences, en_model, fr_model):
    loopcount = len(en_sentences)
    loss_axis = []
    ep_count = 20
    epoch_loss = []
    for epoch in xrange(ep_count):
        for i in xrange(loopcount):
            en_arr = en_sentences[i]  # ["line","one"]
            fr_arr = fr_sentences[i]
            X = sentenceToVector(en_model, en_arr)  # [ [1,2,3], [4,7,1] ]
            Y = sentenceToVector(fr_model, fr_arr)

            if X.ndim < 2 or Y.ndim <2:
                continue
            pred_vector, error = Encoder.train(X, Decoder, Y)
            epoch_loss.append(error)
            # print vectorToSentence(fr_model, pred_vector), "\n----------------\n"
            if(i%2000==0):
                saveWeights()
                # print epoch, i
        loss_axis.append(np.mean(np.square(epoch_loss)))
        epoch_loss = []
        print loss_axis[-1]

    saveWeights()
    plot_graph(len(loss_axis),loss_axis)

def testScript(en_model, fr_model):     #source line, target vector model
    while True:
        s = str(raw_input("enter an english sentence or (exit to quit):"))
        if s == "exit":
            break
        eng_sentence_vec = sentenceToVector(en_model, s.split())
        print vectorToSentence(fr_model, Encoder.predict(eng_sentence_vec, Decoder))


def parse_file(file):
    f= open(file,"r")
    text = f.read()
    lines = text.split("\n")
    lines = lines[0:20000]
    # sentences = [l.split(" ") for l in lines]
    sentences = [filter(None, re.split("[, \-!?:]+", l)) for l in lines]
    f.close()
    return sentences

def prepare_model(en_sentences=[],fr_sentences=[]):
# @params to w2v model
    # size: Denotes the number of dimensions present in the vector form. For sizeable blocks, people use 100-200. We can use around 50 for the Wikipedia articles.
    # window: Only terms hat occur within a window-neighbourhood of a term, in a sentence, are associated with it during training.
    # mincount: terms that occur less than min_count number of times are ignored in the calculations.
    # sg: This defines the algorithm. If equal to 1, the skip-gram technique is used. Else(=0), the CBoW method is employed. By default its 0.
    # workers: to controll parallization. default = 1 = no parallelization

    # my_model = w2v([["line","one"], ["This", "is", "line", "two"]], size=2, window=2, min_count=1, workers=1)
    en_model = w2v(en_sentences, size=50, window=4, min_count=1, workers=4, sg=1)
    fr_model = w2v(fr_sentences, size=50, window=4, min_count=1, workers=4, sg=1)
    en_model.save("models/en_mod")
    fr_model.save("models/fr_mod")
    return en_model,fr_model

def load_model():
    en_model = w2v.load('models/en_mod')
    fr_model = w2v.load('datasets/fr_mod')

    # print en_model.wv.most_similar(positive=[en_model.wv["measurable"]], topn=5)
    # print en_model.wv.most_similar(positive=[en_model.wv["legislation"]], topn=5)
    # print en_model.wv.most_similar(positive=[en_model.wv["drunk"]], topn=5)
    # print en_model.wv.most_similar(positive=[en_model.wv["man"]], topn=5)
    # print en_model.wv.most_similar(positive=[en_model.wv["voting"]], topn=5)

    # print fr_model.wv["(V)."]
    # print fr_model.wv.most_similar(positive=[fr_model.wv["(V)."]], topn=5)
    # print fr_model.wv.most_similar(positive=[fr_model.wv["mesurable"]], topn=5)
    # print fr_model.wv.most_similar(positive=[fr_model.wv["legislation"]], topn=5)
    # print fr_model.wv.most_similar(positive=[fr_model.wv["ivre"]], topn=5)
    # print fr_model.wv.most_similar(positive=[fr_model.wv["homme"]], topn=5)
    # print fr_model.wv.most_similar(positive=[fr_model.wv["vote"]], topn=5)
    return en_model,fr_model

def main():
    start_time = timeit.default_timer()

    en_file = "datasets/corpus.en"#sample_en.txt"
    fr_file = "datasets/corpus.fr"#sample_fr.txt"
    # en_sentences = parse_file(en_file)
    # fr_sentences = parse_file(fr_file)
    # print "eng sentences: ", len(en_sentences)
    # en_model, fr_model = prepare_model(en_sentences,fr_sentences)

    en_model, fr_model = load_model()

    # trainScript(en_sentences, fr_sentences, en_model, fr_model)

    Encoder.set_weights(np.load('save/encoder_weights.npy').item())
    Decoder.set_weights(np.load('save/decoder_weights.npy').item())

    testScript(en_model, fr_model)

    print "ExecutionTime: ", (timeit.default_timer() - start_time)
# ------------------------------------------
main()