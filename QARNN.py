from _gdynet import *
import _gdynet as dy
#from dynet import *
#import dynet as dy
import time
import random
import numpy as np
from sklearn import model_selection

LAYERS = 2
INPUT_DIM = 50 #50  #256
HIDDEN_DIM = 50 # 50  #1024
VOCAB_SIZE = 0

from collections import defaultdict
from itertools import count
import sys
import util


class RNNLanguageModel:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=SimpleRNNBuilder):

        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
        self.lookup = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
        self.R = model.add_parameters((2, HIDDEN_DIM))
        self.bias = model.add_parameters((2))

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
        for i in range(0, len(sent)-1):
            # assume word is already a word-id
            x_t = lookup(self.lookup, int(sent[i]))
            state = state.add_input(x_t)
        #    print "FEEDING", sent[i]
        y_t = state.output()
        r_t = bias + (R * y_t)
        label = int(sent[-1])
        print "sent", sent
        print "LABEL", label
        if label == 37:
            label = 1
        else:
            label = 0
        err = pickneglogsoftmax(r_t, label)
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

    train = util.FastCorpusReaderYahoo("C:\\corpora\\yahoo\\title.csv")

    vocab = util.Vocab.from_corpus(train)

    VOCAB_SIZE = vocab.size()
    model = Model()
    #sgd = SimpleSGDTrainer(model)
    sgd = AdamTrainer(model)
    print ("vocab_size", VOCAB_SIZE)

    lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=LSTMBuilder)
    train = list(train)

    recall_1_list = []
    recall_0_list = []
    loss = 0
    n = len(train)

    kf = model_selection.KFold(n_splits=5)
    for train_idx, test_idx in kf.split(train):

        X_train =  [train[i] for i in train_idx]
        X_test = [train[i] for i in test_idx]
        #TRAIN
        for ITER in xrange(5):  # number of epochs
            for i, sentence in enumerate(X_train):
                print sentence
                isent = [vocab.w2i[w] for w in sentence]
                errs = lm.BuildLMGraph(isent)
                loss += errs.scalar_value()
                errs.backward()
                sgd.update()
                sgd.status()
                sgd.update_epoch()
        #TEST
        correct_0 = 0
        count_0 = 0
        correct_1 = 0
        count_1 = 0
        all_0 = 0
        all_1 = 0
        classified_as_1 = 0
        classified_as_0 = 0

        for sentence in X_test:
            isent = [vocab.w2i[w] for w in sentence]
            sent = isent[0:len(isent) - 1]
            label = isent[-1]
            probs = lm.predictNextWord(isent)
            distribution = probs.npvalue()
            answer = np.argmax(distribution)

            if answer == 0 and label == 10:
                correct_0 += 1

            if answer == 1 and label == 37:
                correct_1 += 1

            if answer == 0:
                all_0 += 1
            else:
                all_1 += 1

            if label == 37:
                count_1 += 1
            else:
                count_0 += 1
            print sent, label, answer
        recall_1_list.append(correct_1 / float(count_1))
        recall_0_list.append(correct_1 / float(count_0))

    print "RECALL 1 list:", recall_1_list
    print "RECALL 0 list:", recall_0_list

    print sum(recall_1_list) / float(len(recall_1_list))
    print sum(recall_0_list) / float(len(recall_0_list))

    '''
    train_t = train[0:800]
    test_t = train[800:1000]

    train_arr = []
    test_arr = []

    train_arr.append(train[0:800])
    test_arr.append(train[801:1000])

    train_arr.append(train[0:800])
    test_arr.append(train[801:1000])

    length = loss = 0.0

    start_time = time.time()



    for ITER in xrange(5): #number of epochs

        for i, sentence in enumerate(train_t):
            print sentence
            length += len(sentence)-1
            isent = [vocab.w2i[w] for w in sentence]
            errs = lm.BuildLMGraph(isent)
            loss += errs.scalar_value()
            errs.backward()
            sgd.update()
            sgd.status()
            sgd.update_epoch()

        lm.save2disk("C:\\corpora\\yahoo.txt")

    print("--- %s seconds ---" % (time.time() - start_time))

    lm.load_from_disk("C:\\corpora\\yahoo.txt")

    #37=1
    #0=10

    correct_0 = 0
    count_0 =0
    correct_1 = 0
    count_1 = 0

    all_0 = 0
    all_1 = 0


    classified_as_1 = 0
    classified_as_0 = 0

    for sentence in test_t:
        isent = [vocab.w2i[w] for w in sentence]
        sent = isent[0:len(isent)-1]
        label = isent[-1]
        probs = lm.predictNextWord(isent)
        distribution = probs.npvalue()
        answer = np.argmax(distribution)

        if answer == 0 and label == 10:
            correct_0 +=1

        if answer == 1 and label == 37:
            correct_1 +=1

        if answer == 0:
            all_0 +=1
        else:
            all_1 +=1

        if label==37:
            count_1 +=1
        else:
            count_0 +=1
        print sent, label, answer

    print "RECALL 1:", correct_1 / float(count_1)
    print "RECALL 0:", correct_1 / float(count_0)
    print "Precision 1:", correct_1 / float(all_1)
    print "Precision 0:", correct_0 / float(all_0)
'''