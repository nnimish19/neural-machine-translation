import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec as w2v
import encoder as encoder
import decoder as decoder
from sklearn.metrics.pairwise import cosine_similarity as cs

Encoder = encoder.RNNClassifier()
Decoder = decoder.RNNClassifier()

def plot_graph(n, loss_axis):
    # print loss_axis
    plt.xlabel("# ith Example", fontsize=15)
    plt.ylabel("Training Error", fontsize=15)
    plt.plot(np.arange(n), loss_axis, '-')
    plt.show()

def sentenceToVector(model,s,word_map):     #Array of words
    list= [] #list of 2D np-array
    for w in s:
        if w in word_map:
            list.append(model[word_map.index(w)])
        else:
            list.append(model[word_map.index("<unk>")])

    return np.array(list)

def vectorToSentence(matrix,map,V):    #Array of word_embeddings
    s=""
    dim = len(matrix)
    for w in V:
        cosine_list = list(np.asarray(cs(matrix,w)).flatten())
        s = s + " " + map[cosine_list.index(max(cosine_list))]
    return s

def trainScript(en_sentences, fr_sentences, en_model, fr_model,en_map,fr_map):
    loopcount = len(en_sentences)
    loss_axis = []
    ep = 100
    Encoder_weights = {'W2': [], 'b2':[],'Wh':[],'W1':[], 'b1':[]}
    Decoder_weights = {'W2': [], 'b2': [], 'Wh': [], 'W1': [], 'b1': [],'dWy':[]}

    for epoch in xrange(ep):
        epoch_loss = []
        for i in xrange(loopcount):
            en_arr = en_sentences[i]  # ["line","one"]
            fr_arr = fr_sentences[i]
            X = sentenceToVector(en_model, en_arr, en_map)  # [ [1,2,3], [4,7,1] ]
            Y = sentenceToVector(fr_model, fr_arr, fr_map)
            pred_vector, error = Encoder.train(X, Decoder, Y)
            epoch_loss.append(error)
            # print vectorToSentence(fr_model, pred_vector), "\n----------------\n"
        loss_axis.append(np.mean(np.square(epoch_loss)))

    Encoder_weights['W2'], Encoder_weights['b2'], Encoder_weights['Wh'], Encoder_weights['W1'],Encoder_weights['b1'] = Encoder.get_weights()
    Decoder_weights['dW2'],Decoder_weights['db2'],Decoder_weights['dWh'],Decoder_weights['dWy'],Decoder_weights['dW1'],Decoder_weights['db1'] = Decoder.get_weights()
    np.save('encoder_weights.npy', Encoder_weights)
    np.save('decoder_weights.npy', Decoder_weights)
    plot_graph(ep,loss_axis)

def testScript(vec, fr_model,fr_map):     #source line, target vector model
    print vectorToSentence(fr_model,fr_map,Encoder.predict(vec, Decoder))


def parse_file(file):
    f= open(file,"r")
    text = f.read()
    lines = text.split("\n")
    sentences = [l.split(" ") for l in lines]
    f.close()
    return sentences

def prepare_model(en_sentences,fr_sentences):
    # model = w2v([["line","one"], ["This", "is", "line", "two"]], size=2, window=2, min_count=1, workers=1)
    #en_model = w2v(en_sentences, size=5, window=5, min_count=1, workers=4)
    #fr_model = w2v(fr_sentences, size=5, window=5, min_count=1, workers=4)
    en_model = np.loadtxt("C:\Users\\vivek\Documents\Machine-translation-project\datasets\en_matrix.txt")
    fr_model = np.loadtxt("C:\Users\\vivek\Documents\Machine-translation-project\datasets\\fr_matrix.txt")
    en_word_map = list(np.loadtxt("C:\Users\\vivek\Documents\Machine-translation-project\datasets\en_word_vector.txt", dtype=str))
    fr_word_map = list(np.loadtxt("C:\Users\\vivek\Documents\Machine-translation-project\datasets\\fr_word_vector.txt", dtype=str))

    return en_model,fr_model,en_word_map,fr_word_map
    # model.save(fname)
    # en_model = w2v.load('datasets/english.bin')
    # fr_model = w2v.load('datasets/french.bin')

def main():
    en_file = "datasets/en_sample.txt"
    fr_file = "datasets/fr_sample.txt"
    en_sentences = parse_file(en_file)
    fr_sentences = parse_file(fr_file)
    en_model, fr_model, en_word_map, fr_word_map = prepare_model(en_sentences,fr_sentences)

    #trainScript(en_sentences, fr_sentences, en_model, fr_model, en_word_map, fr_word_map)


    Encoder.set_weights(np.load('encoder_weights.npy').item())
    Decoder.set_weights(np.load('decoder_weights.npy').item())

    strng = ""
    while True:
        strng = str(raw_input("enter an english sentence or (exit to quit):"))
        if strng == "exit":
            break
        eng_sentence_vec = sentenceToVector(en_model, strng.split(), en_word_map)
        testScript(eng_sentence_vec,fr_model,fr_word_map)

main()