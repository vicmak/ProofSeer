from _gdynet import *
import _gdynet as dy
#from dynet import *
#import dynet as dy
import time
import random
import numpy as np


LAYERS = 2
INPUT_DIM = 200 #50  #256
HIDDEN_DIM = 300 # 50  #1024
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

    def save2disk(self, filename):
        model.save(filename, [self.builder, self.lookup, self.R, self.bias])

    def load_from_disk(self, filename):
        (self.builder, self.lookup, self.R, self.bias) = model.load(filename)

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
        prob = softmax(r_t)
        return prob


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


def log_train_file(message, error):
    log_file = "C:\\corpora\\log.txt"
    logline = message + " " + str(error) + "\n"
    with open(log_file, "a") as myfile:
        myfile.write(logline)


if __name__ == '__main__':

    dy.init()

    print "DyNet was initialized, starting train"

    train = util.FastCorpusReader("C:\\corpora\\long30k.txt")

    vocab = util.Vocab.from_corpus(train)

    VOCAB_SIZE = vocab.size()
    model = Model()
    sgd = SimpleSGDTrainer(model)
    print ("vocab_size", VOCAB_SIZE)

    lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=LSTMBuilder)
    train = list(train)

    length = loss = 0.0

    start_time = time.time()

    for ITER in xrange(4): #number of epochs

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
        lm.save2disk("C:\\corpora\\back_bigmodel.txt")
        log_train_file("backward epoch" + str(ITER), loss)



    print("--- %s seconds ---" % (time.time() - start_time))
    '''
    start_sentence = ['<start>', 'with', 'hardly', 'a', 'word']
    isent = [vocab.w2i[w] for w in start_sentence]
    print "ISENT",isent
    distribution = lm.predictNextWord(isent).npvalue()

    for i in range(0, 20):
        max_index = np.argmax(distribution)
        print i, vocab.w2i.keys()[vocab.w2i.values().index(max_index)]
        distribution = np.delete(distribution, max_index)

    #lm.save2disk("C:\\Users\\makarenk\\Downloads\\bigmodel.txt")

    print "loading model"
    lm.load_from_disk("C:\\corpora\\bigmodel.txt")
    print "here 111"
    start_sentence = ['<start>', 'the', 'results', 'clearly']
    isent = [vocab.w2i[w] for w in start_sentence]
    print "ISENT", isent
    distribution = lm.predictNextWord(isent).npvalue()

    for i in range(0, 20):
        max_index = np.argmax(distribution)
        print i, vocab.w2i.keys()[vocab.w2i.values().index(max_index)]
        distribution = np.delete(distribution, max_index)

    start_sentence = ['<start>', 'query', 'performance']
    isent = [vocab.w2i[w] for w in start_sentence]
    print "ISENT", isent
    distribution = lm.predictNextWord(isent).npvalue()

    for i in range(0, 20):
        max_index = np.argmax(distribution)
        print i, vocab.w2i.keys()[vocab.w2i.values().index(max_index)]
        distribution = np.delete(distribution, max_index)

    start_sentence = ['<start>', 'second', 'we', 'present', 'experimental', 'results']
    isent = [vocab.w2i[w] for w in start_sentence]
    print "ISENT", isent
    distribution = lm.predictNextWord(isent).npvalue()

    for i in range(0, 20):
        max_index = np.argmax(distribution)
        print i, vocab.w2i.keys()[vocab.w2i.values().index(max_index)]
        distribution = np.delete(distribution, max_index) '''
