from _gdynet import *
import _gdynet as dy
#from dynet import *
#import dynet as dy
import time
import random
import numpy as np
from sklearn import model_selection
from sklearn.metrics import roc_auc_score

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

    def build_sentence_graph(self, sents, labels):
        renew_cg()
        f_init = self.builder.initial_state()

        STOP = vocab.w2i["<stop>"]
        START = vocab.w2i["<start>"]

        W_exp = parameter(self.R)
        b_exp = parameter(self.bias)
        state = f_init

        # get the wids and masks for each step
        tot_words = 0
        wids = []
        masks = []

        for i in range(len(sents[0])):
            wids.append([(START if len(sents[0])-len(sent) > i else vocab.w2i[sent[i - len(sents[0])+len(sent)]]) for sent in sents])
            #wids.append([(vocab.w2i[sent[i]] if len(sent) > i else STOP) for sent in sents])
            mask = [(1 if len(sent) > i else 0) for sent in sents]
            masks.append(mask)
            tot_words += sum(mask)

        #print "wids:"
        #print wids

        # start the rnn by inputting "<start>"
        init_ids = [START] * len(sents)
        s = f_init.add_input(dy.lookup_batch(self.lookup, init_ids))

        # feed word vectors into the RNN and predict the next word
        losses = []
        for wid in wids:
            # calculate the softmax and loss
            score = W_exp * s.output() + b_exp
            # update the state of the RNN
            wemb = dy.lookup_batch(self.lookup, wid)
            s = s.add_input(wemb)

        loss = dy.pickneglogsoftmax_batch(score, labels)

        losses.append(loss)
        return dy.sum_batches(dy.esum(losses))


    def predict_class(self, sentence):
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

def readY(fname):
    train = []
    with file(fname) as fh:
        for line in fh:
            line = line.lower()
            train.append(int(line.strip()[-1]))
    return train

if __name__ == '__main__':

    MB_SIZE = 100

    filename = "C:\\corpora\\yahoo\\Title_3.csv"
    train = util.FastCorpusReaderYahoo(filename)
    vocab = util.Vocab.from_corpus(train)

    Ys = readY(filename)
    train = list(train)

    print len(train), len(Ys)

    batches_num = len(train) // MB_SIZE

    train = train[0: MB_SIZE*batches_num]
    Ys = Ys[0: MB_SIZE*batches_num]

    print len(train), len(Ys)

    data = zip(train, Ys)

    data.sort(key=lambda x: -len(x[0]))

    x2, y2 = zip(*data)

    train = list(x2)
    Ys = list(y2)

    #for i in range(0, len(train)):
    #    print train[i], Ys[i]

    for i in range(0, len(train)):
        print train[i], Ys[i]

    VOCAB_SIZE = vocab.size()
    print ("vocab_size", VOCAB_SIZE)

    dy.init()
    print "DyNet was initialized, starting train"


    recall_1_list = []
    recall_0_list = []
    loss = 0
    n = len(train)
    auc = []

    kf = model_selection.KFold(n_splits=5)
    for train_idx, test_idx in kf.split(train):

        model = Model()
        sgd = AdamTrainer(model)
        lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=LSTMBuilder)


        X_train =  [train[i] for i in train_idx]
        Y_train = [Ys[i] for i in train_idx]

        train_order = [x * MB_SIZE for x in range(len(X_train) / MB_SIZE)]

        X_test = [train[i] for i in test_idx]
        Y_test = [Ys[i] for i in test_idx]
        #TRAIN
        for ITER in xrange(3):  # number of epochs to pass all data

            for i, sid in enumerate(train_order, 1):
                print "Batch: ", i, " of ", len(train) / MB_SIZE, "epoch: ", str(ITER)
                loss_exp = lm.build_sentence_graph(X_train[sid:sid + MB_SIZE], Y_train[sid:sid + MB_SIZE])
                loss += loss_exp.scalar_value()
                loss_exp.backward()
                sgd.update()
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

        res = []
        for i, sentence in enumerate(X_test):
            isent = [vocab.w2i[w] for w in sentence]
            sent = isent[0:len(isent) - 1]
            label = Y_test[i]
            probs = lm.predict_class(isent)
            distribution = probs.npvalue()
            answer = np.argmax(distribution)
            res.append(answer)
            if answer == 0 and label == 0:
                correct_0 += 1

            if answer == 1 and label == 1:
                correct_1 += 1

            if answer == 0:
                all_0 += 1
            else:
                all_1 += 1

            if label == 1:
                count_1 += 1
            else:
                count_0 += 1
            print sent, label, answer

        auc.append(roc_auc_score(Y_test, res))
        recall_1_list.append(correct_1 / float(count_1))
        recall_0_list.append(correct_1 / float(count_0))

    print "RECALL 1 list:", recall_1_list
    print "RECALL 0 list:", recall_0_list

    print "RECALL 0:", sum(recall_0_list) / float(len(recall_0_list))
    print "RECALL 1:", sum(recall_1_list) / float(len(recall_1_list))
    print "AUC :", sum(auc)/float(len(auc))
    ''' '''