from _gdynet import *
import _gdynet as dy
import numpy as np
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
import mmap
import nltk
from collections import defaultdict
from itertools import count

LAYERS = 2
INPUT_DIM = 50 #50  #256
HIDDEN_DIM = 50 # 50  #1024
VOCAB_SIZE = 0

class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

def ExtractAlphanumeric(ins):
    from string import ascii_letters, digits, whitespace, punctuation
    return "".join([ch for ch in ins if ch in (ascii_letters + digits + whitespace + punctuation)])

class FastCorpusReaderYahoo:
    def __init__(self, fname):
        self.fname = fname
        self.f = open(fname, 'rb')

    def __iter__(self):
        #in Linux\Mac replace with m = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
        m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        data = m.readline()
        while data:
            parts = data.split(",")
            end = len(parts)-1
            text_parts = parts[1 : end]
            line = ",".join(text_parts)
            data = m.readline()
            line = line.lower()
            line = ExtractAlphanumeric(line)
            tokens = nltk.word_tokenize(line)
            line = ["<start>"] + tokens + ["<stop>"]
            yield line


class RNN_sequence_model:
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

        START = vocab.w2i["<start>"]

        W_exp = parameter(self.R)
        b_exp = parameter(self.bias)
        state = f_init

        # get the wids and masks for each step
        tot_words = 0
        wids = []
        masks = []

        #pad the sequences that are shorter, with a START at the beginning
        for i in range(len(sents[0])):
            wids.append([(START if len(sents[0])-len(sent) > i else vocab.w2i[sent[i - len(sents[0])+len(sent)]]) for sent in sents])
            mask = [(1 if len(sent) > i else 0) for sent in sents]
            masks.append(mask)
            tot_words += sum(mask)

        init_ids = [START] * len(sents)
        s = f_init.add_input(dy.lookup_batch(self.lookup, init_ids))

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
    train = FastCorpusReaderYahoo(filename)
    vocab = Vocab.from_corpus(train)

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
        lm = RNN_sequence_model(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=LSTMBuilder)


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