import numpy as np 
import pandas as pd


def word_2_vector(data,fr_obj):
    com_vec = []
    for line in data:
        v = []
        l = line.rstrip().split(" ")
        for wr in l:
            try:
               rank = fr_obj[wr] 
               v.append(rank)            
            except KeyError:
                continue
        com_vec.append(v)
    return com_vec

    

if __name__ == "__main__":
    eng_data= open("/home/vivekgade/Desktop/Machine Learning/Project/data set/english/english corpus.en")
    eng_data = eng_data.readlines()
    fr_data= open("/home/vivekgade/Desktop/Machine Learning/Project/data set/french/french corpus.fr")
    fr_data = fr_data.readlines()
    eng_frq_lst = pd.read_csv("/home/vivekgade/Desktop/Machine Learning/Project/data set/english/English frequency list.csv")
    eng_frq_obj = dict(zip(eng_frq_lst.Word,eng_frq_lst.Rank)) 
    fr_frq_lst =  pd.read_csv("/home/vivekgade/Desktop/Machine Learning/Project/data set/french/French frequency list.csv")
    fr_frq_obj = dict(zip(fr_frq_lst.word,fr_frq_lst['rank']))
    
    #eng_w2v_list = word_2_vector(eng_data,eng_frq_obj) #english sentence wise vector
    
    #fr_w2v_list = word_2_vector(eng_data,eng_frq_obj) #french sentence wise vector