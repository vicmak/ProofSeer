from dynet import *
import time
import random
import numpy as np

LAYERS = 2
INPUT_DIM = 100 #50  #256
HIDDEN_DIM = 256 # 50  #1024
VOCAB_SIZE = 0

from collections import defaultdict
from itertools import count
import sys
import util


class RNNLanguageModel:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=SimpleRNNBuilder):
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        self.lookup = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
        self.R = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
        self.bias = model.add_parameters((VOCAB_SIZE))

    def BuildLMGraph(self, sent):
        renew_cg()
        init_state = self.builder.initial_state()

        R = parameter(self.R)
        bias = parameter(self.bias)
        errs = [] # will hold expressions
        es=[]
        state = init_state
        for (cw, nw) in zip(sent,sent[1:]):
            # assume word is already a word-id
            x_t = lookup(self.lookup, int(cw))
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        nerr = esum(errs)
        return nerr


    def predictNextWord(self, sentence):
        renew_cg()
        init_state = self.builder.initial_state()
        R = parameter(self.R)
        bias = parameter(self.bias)
        state = init_state
        for cw in sentence:

            # assume word is already a word-id
            x_t = lookup(self.lookup, int(cw))
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = softmax(r_t)
        return err


    def sample(self, first=1, nchars=0, stop=-1):
        res = [first]
        renew_cg()
        state = self.builder.initial_state()

        R = parameter(self.R)
        bias = parameter(self.bias)
        cw = first
        while True:
            x_t = lookup(self.lookup, cw)
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            ydist = softmax(r_t)
            dist = ydist.vec_value()
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res

if __name__ == '__main__':
    #train = util.CharsCorpusReader(sys.argv[1], begin="<s>")
    train = util.CorpusReader("/Users/macbook/Desktop/corpora/big2.txt")

    vocab = util.Vocab.from_corpus(train)

    VOCAB_SIZE = vocab.size()

    model = Model()
    sgd = SimpleSGDTrainer(model)
    print ("vocab_size", VOCAB_SIZE)
    #lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=SimpleRNNBuilder)
    lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=LSTMBuilder)
    train = list(train)
    print train

    length = loss = 0.0
    for ITER in xrange(100): #number of epochs

        random.shuffle(train)
        for i, sentence in enumerate(train):
            print (i, sentence)

            length += len(sentence)-1
            #print ("Sentence", sentence)
            #print ("length", length)
            isent = [vocab.w2i[w] for w in sentence]
            #print ("isent", isent) #isent is the sentence represented in numbers from vocab
            errs = lm.BuildLMGraph(isent)
            loss += errs.scalar_value()
            #print ("Loss:", loss)
            errs.backward()
            sgd.update(1.0)
            sgd.status()
            sgd.update_epoch(1.0)

    start_sentence = ['<start>', 'with', 'hardly', 'a', 'word']
    isent = [vocab.w2i[w] for w in start_sentence]
    print "ISENT",isent
    distribution = lm.predictNextWord(isent).npvalue()

    for i in range(0, 20):
        max_index = np.argmax(distribution)
        print i, vocab.w2i.keys()[vocab.w2i.values().index(max_index)]
        distribution = np.delete(distribution, max_index)

    print lookup(lm.lookup, 5).value()