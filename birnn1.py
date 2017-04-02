from collections import Counter, defaultdict
from itertools import count
import random
import time
import _gdynet as dy
from _gdynet import *
import numpy as np
import nltk
from nltk.corpus import wordnet as wn

# format of files: each line is "word<TAB>tag<newline>", blank line is new sentence.
train_file = "C:\\corpora\\long30k.txt"
test_file = "C:\\corpora\\test.txt"


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


def read(fname):

    count = 0
    with file(fname) as fh:
        for line in fh:
            count = count + 1
            sent = line.lower().strip().split()
            #sent.append("<stop>")
            sent = ["<start>"] + sent + ["<stop>"]
            #sent.reverse()
            if count <= 2775000 and len(sent) <=42:
                yield sent


train = list(read(train_file))
test = list(read(test_file))
words = []
wc = Counter()
for sent in train:
    for w in sent:
        words.append(w)
        wc[w] += 1

vw = Vocab.from_corpus([words])
STOP = vw.w2i["<stop>"]
START = vw.w2i["<start>"]
nwords = vw.size()

LAYERS = 1
INPUT_DIM = 200 #50  #256
HIDDEN_DIM = 200 # 50  #1024
print "words", nwords

dy.init()
model = dy.Model()
sgd = dy.AdamTrainer(model)

WORDS_LOOKUP = model.add_lookup_parameters((nwords, INPUT_DIM))

W_sm = model.add_parameters((nwords, HIDDEN_DIM * 2))
b_sm = model.add_parameters(nwords)

builders = [
        LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model),
        LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model),
        ]

def predict_middle_word(iprefix, ipostfix, builders):
    renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]
    W = parameter(W_sm)
    b = parameter(b_sm)
    prefix_wembs = [lookup(WORDS_LOOKUP, int(w)) for w in iprefix]
    postfix_wembs = [lookup(WORDS_LOOKUP, int(w)) for w in ipostfix]

    state = f_init
    for i in range(len(prefix_wembs) - 1):
        # assume word is already a word-embedding
        x_t = prefix_wembs[i]
        state = state.add_input(x_t)
    prefix_y_t = state.output()

    state = b_init
    for i in range(len(postfix_wembs) - 1):
        x_t = postfix_wembs[i]
        state = state.add_input(x_t)
    postfix_y_t = state.output()

    y = concatenate([prefix_y_t, postfix_y_t])
    r = b + (W * y)
    prob = softmax(r)
    return prob

def evaluate_sentence(sentence, vw, model1):

    print sentence
    mid_index = get_check_index(sentence)
    prefix = get_prefix(sentence, mid_index)
    postfix = get_postfix(sentence, mid_index)
    original_word = get_original_word(sentence, mid_index)
    corrected_word = get_corrected_word(sentence, mid_index)

    tags = nltk.pos_tag(prefix + [original_word])
    original_word_tag = tags[-1][1]
    prefix = ["<start>"] + prefix + [original_word]


    postfix.reverse()
    postfix = ["<stop>"] + postfix + [original_word]

    predictions = get_predictions(prefix, postfix, vw, model1, original_word, original_word_tag)
    mrr = get_mrr(predictions, corrected_word)
    log_sentence(sentence, predictions, mrr)
    return mrr

def get_predictions(prefix, postfix, vw, model1, original_word, original_word_tag):

    print prefix
    print postfix
    print original_word
    print original_word_tag

    iprefix = [vw.w2i[w] for w in prefix if w in vw.w2i.keys()]
    ipostfix = [vw.w2i[w] for w in postfix if w in vw.w2i.keys()]

    predictions = []
    wordnet_tag = get_wordnet_pos_code(original_word_tag)
    if (len(prefix) > 1 and len(postfix) > 1):
        distribution = predict_middle_word(iprefix, ipostfix, model1[0]).npvalue()

        predictions_indexes = np.argsort(distribution)[-100:]
        predictions_indexes = predictions_indexes[::-1]
        for max_index in predictions_indexes:
            prediction = vw.w2i.keys()[vw.w2i.values().index(max_index)]
            if len(wn.synsets(prediction, wordnet_tag)) > 0 or wordnet_tag == '':
                predictions.append(vw.w2i.keys()[vw.w2i.values().index(max_index)])

    return predictions

def get_mrr(predictions, correct_word):
    rank = 1
    start_index = 0

    while start_index < len(predictions):
        if correct_word != predictions[start_index]:
            rank = rank + 1
        else:
            break
        start_index = start_index + 1

    if start_index == len(predictions):
        return 0
    return 1.0/float(rank)


def log_sentence(sentence, predictions, mrr):
    filename = "c:\\corpora\\evaluation.txt"
    with open(filename, "a") as log_file:
        log_file.write(" ".join(sentence) + "\n")
        log_file.write("MRR:" + str(mrr) + "\n")
        log_file.write("Predictions: \n" + ",".join(predictions)  +"\n \n")

def clean_text(token):
    token = token.replace(",", "")
    token = token.replace("-", " ")
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

def get_wordnet_pos_code(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('VB'):
        return wn.VERB
    elif tag.startswith('JJ'):
        return wn.ADJ
    elif tag.startswith('RB'):
        return wn.ADV
    else:
        return ''


def build_sentence_graph(isent, builders):
    renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]
    W = parameter(W_sm)
    b = parameter(b_sm)

    wembs = [lookup(WORDS_LOOKUP, int(w)) for w in isent]

    forward_y_ts = []
    state = f_init
    for i in range(len(wembs)-1):
        # assume word is already a word-embedding
        x_t = wembs[i]
        state = state.add_input(x_t)
        y_t = state.output()
        forward_y_ts.append(y_t)

    state = b_init
    backward_y_ts = []
    for i in range(len(wembs) - 1):
        # assume word is already a word-embedding
        x_t = wembs[::-1][i]
        state = state.add_input(x_t)
        y_t = state.output()
        backward_y_ts.append(y_t)

    errs = []

    for i in range(len(wembs) - 3):
        y = concatenate([forward_y_ts[i], backward_y_ts[::-1][i+2]])
        r = b + (W * y)
        err = pickneglogsoftmax(r, int(isent[i+1]))
        errs.append(err)

    nerr = esum(errs)

    return nerr


train_size = len(train)
loss = 0
for ITER in range(50):
    random.shuffle(train)
    for i, sentence in enumerate(train, 1):
        if i % 50 == 0:
            sgd.status()
            print "Loss", loss, "Iter", ITER, "i:",i, "of ", train_size
            loss = 0
        isent = [vw.w2i[w] for w in sentence]
        sum_errs = build_sentence_graph(isent, builders)
        loss += sum_errs.scalar_value()
        sum_errs.backward()
        sgd.update()

    model.save("C:\\corpora\\adam_bilstm_bigmodel.txt", [builders[0], builders[1], WORDS_LOOKUP, W_sm, b_sm])
'''

sents = load_sentences("c:\\corpora\\corrected_replace.txt")
(builders[0], builders[1], WORDS_LOOKUP, W_sm, b_sm) = model.load("C:\\corpora\\adam_bilstm_bigmodel.txt")
total_mrr = 0
for sentence in sents:
    total_mrr = total_mrr + evaluate_sentence(sentence, vw, [builders, WORDS_LOOKUP, W_sm, b_sm])
print "Total Mrr", total_mrr/float(len(sents))
'''