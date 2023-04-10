import random
import scipy.io
import math
from scipy.stats import binom

def Probabilistic_Method2(unique_words): 
    h = 2.2*len(unique_words)/(math.log(len(unique_words)))
    return  math.ceil(h)

def Random_Resolving_Set_Compliment(unique_words):
    h = Probabilistic_Method2(unique_words)
    #print(h)
    # dictionary = dict.fromkeys( (range(1,len(unique_words))), unique_words)
    #print(dictionary)
    dictionary = dict()
    #print(len(unique_words))
    unique_words_poppy = unique_words
    for i in range(1,len(unique_words_poppy)):
        #dictionary = dict.fromkeys(unique_words.pop(), i)
        dictionary[i] = unique_words_poppy.pop()
    r = []
    q = []
    #print(len(dictionary))
    for i in range(1, h):
        Rlen = binom.rvs(len(dictionary), .5)
        for x in range(0,Rlen):
            k = random.randint(1, len(dictionary))
            r.append(dictionary.get(k))
        q.append(r)
        q.append(list(set(dictionary.values())-set(r)))
        r = []
    return(q)

def JVecEMatrix(resolving,tweets):
    veclist = []
    for tweet in tweets:
        veclist.append(JacVector(resolving,tweet))
    #q = scipy.spatial.distance.pdist(veclist, metric='euclidean')
    return veclist

def JacVector(resolving, test):
    M = []
    for i in range(len(resolving)):
            M.append(JaccardSim(set(resolving[i]),set(test)))
    return(M)

def JaccardSim(a,b):
    U = a.union(b)
    I = a.intersection(b)
    similarity = (len(I)/len(U))
    return similarity


def Random_Resolving_Set(unique_words):
    h = Probabilistic_Method2(unique_words)
    dictionary = dict()
    unique_words_poppy = unique_words
    for i in range(1,len(unique_words_poppy)):
        dictionary[i] = unique_words_poppy.pop()
    r = []
    q = []
    for i in range(1, h):
        Rlen = binom.rvs(len(dictionary), .5)
        for x in range(0,Rlen):
            k = random.randint(1, len(dictionary))
            r.append(dictionary.get(k))
        q.append(r)
        r = []
    return(q)