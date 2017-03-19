from collections import Counter, defaultdict
from itertools import count
import random
import time
import _gdynet as dy
import numpy as np


class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.iteritems()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

def read(fname):
    """
    Read a file where each line is of the form "word1 word2 ..."
    Yields lists of the form [word1, word2, ...]
    """
    count = 0
    with file(fname) as fh:
        for line in fh:
            count = count + 1
            sent = line.lower().strip().split()
            sent = ["<start>"] + sent + ["<stop>"]
            #sent.reverse()
            if count <= 2775000 and len(sent) <=42:
                yield sent

def clean_text(token):
    token = token.replace(",", "")
    token = token.replace("-", "")
    token = token.replace(".", "")
    token = token.replace("\'", "")
    token = token.replace("?", "")
    token = token.replace(":", "")
    token = token.replace(";", "")
    token = token.replace("\"", "")
    token = token.replace(")", "")
    token = token.replace("(", "")
    token = token.replace("[", "")
    token = token.replace("]", "")
    token = token.replace("}", "")
    token = token.replace("{", "")
    return token


def load_sentences(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = clean_text(line)
            line = line.lower().strip()
            tokens = line.split()
            sents.append(tokens)
           #print tokens
    return sents


def get_check_index(sent):
    for i in xrange(len(sent)):
        if '*' in sent[i]:
            return i
    return -1


def get_prefix(sent, mid_index):
    prefix = []
    for i in xrange(len(sent)):
        if i < mid_index:
            prefix.append(sent[i])
    return prefix


def get_postfix(sent, mid_index):
    postfix = []
    for i in xrange(len(sent)):
        if i > mid_index:
            postfix.append(sent[i])
    return postfix


def get_original_word(sent, mid_index):
    mid_word = sent[mid_index]
    split_words = mid_word.split("/")
    original_word = split_words[0][1:len(split_words[0])]
    return original_word


def get_corrected_word(sent, mid_index):
    mid_word = sent[mid_index]
    split_words = mid_word.split("/")
    corrected_word = split_words[1]
    return corrected_word


def get_predictions(prefix, postfix, vw, model1):
    prefix = ["<start>"] + prefix
    isent = [vw.w2i[w] for w in prefix if w in vw.w2i.keys()]
    print prefix, isent
    predictions = []
    if (len(prefix) > 1):
        distribution = predictNextWord(isent, model1[0], model1[1], model1[2], model1[3]).npvalue()
        for i in range(0, 20):
            max_index = np.argmax(distribution)
            predictions.append(vw.w2i.keys()[vw.w2i.values().index(max_index)])
            distribution = np.delete(distribution, max_index)
    return predictions


def get_mrr(predictions, correct):
    return 1


def log_sentence(sentence, predictions, mrr):
    filename = "c:\\corpora\\evaluation.txt"
    with open(filename, "a") as log_file:
        log_file.write(" ".join(sentence) + "\n")
        log_file.write("MRR:" + str(mrr) + "\n")
        log_file.write("Predictions: " + ",".join(predictions) + "\n \n \n")


def evaluate_sentence(sentence, vw, model1):

    mid_index = get_check_index(sentence)
    prefix = get_prefix(sentence, mid_index)
    postfix = get_postfix(sentence, mid_index)
    original_word = get_original_word(sentence, mid_index)
    corrected_word = get_corrected_word(sentence, mid_index)
    predictions = get_predictions(prefix, postfix, vw, model1)
    mrr = get_mrr(predictions, corrected_word)
    log_sentence(sentence, predictions, mrr)


def predictNextWord(sentence, builder, wlookup, mR, mB):
    dy.renew_cg()
    init_state = builder.initial_state()
    R = dy.parameter(mR)
    bias = dy.parameter(mB)
    state = init_state
    for cw in sentence:
        # assume word is already a word-id
        x_t = dy.lookup(wlookup, int(cw))
        state = state.add_input(x_t)
    y_t = state.output()
    r_t = bias + (R * y_t)
    prob = dy.softmax(r_t)
    return prob


def main():
    print "Starting evaluation..."
    sents = load_sentences("c:\\corpora\\corrected.txt")
    for i in xrange(len(sents)):
        mid_index = get_check_index(sents[i])
        print get_original_word(sents[i], mid_index), "->", get_corrected_word(sents[i], mid_index)

    train = list(read("C:\\corpora\\long30k.txt"))
    words = []
    wc = Counter()
    for sent in train:
        for w in sent:
            words.append(w)
            wc[w] += 1

    vw = Vocab.from_corpus([words])

    nwords = vw.size()
    LAYERS = 2
    INPUT_DIM = 200  # 50  #256
    HIDDEN_DIM = 300  # 50  #1024
    print "words", nwords
    # DyNet Starts
    dy.init()
    model = dy.Model()
    #W_sm = model.add_parameters((nwords, HIDDEN_DIM))
    #b_sm = model.add_parameters(nwords)
    #trainer = dy.SimpleSGDTrainer(model)
    #WORDS_LOOKUP = model.add_lookup_parameters((nwords, INPUT_DIM))
    #RNN = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
    (RNN, WORDS_LOOKUP, W_sm, b_sm) = model.load("C:\\corpora\\batch_bigmodel.txt")

    for sentence in sents:
        evaluate_sentence(sentence, vw, [RNN, WORDS_LOOKUP, W_sm, b_sm])

if __name__ == "__main__":
    main()