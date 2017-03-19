from collections import Counter, defaultdict
from itertools import count
import random
import time
import _gdynet as dy
import numpy as np

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file = "C:\\corpora\\long30k.txt"
test_file = "C:\\corpora\\long30k.txt"

MB_SIZE = 100

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
    """
    Read a file where each line is of the form "word1 word2 ..."
    Yields lists of the form [word1, word2, ...]
    """
    count = 0
    with file(fname) as fh:
        for line in fh:
            count = count + 1
            sent = line.lower().strip().split()
            #sent.append("<stop>")
            sent = ["<start>"] + sent + ["<stop>"]
            sent.reverse()
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

LAYERS = 2
INPUT_DIM = 200 #50  #256
HIDDEN_DIM = 300 # 50  #1024
print "words", nwords
# DyNet Starts
dy.init()
model = dy.Model()
#trainer = dy.AdamTrainer(model)
trainer = dy.SimpleSGDTrainer(model)
# Lookup parameters for word embeddings
WORDS_LOOKUP = model.add_lookup_parameters((nwords, INPUT_DIM))

# Word-level LSTM (layers=1, input=64, output=128, model)


RNN = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

# Softmax weights/biases on top of LSTM outputs
W_sm = model.add_parameters((nwords, HIDDEN_DIM))
b_sm = model.add_parameters(nwords)

# Build the language model graph
def calc_lm_loss(sents):

    dy.renew_cg()
    # parameters -> expressions
    W_exp = dy.parameter(W_sm)
    b_exp = dy.parameter(b_sm)

    # initialize the RNN
    f_init = RNN.initial_state()

    # get the wids and masks for each step
    tot_words = 0
    wids = []
    masks = []

    for i in range(len(sents[0])):
        wids.append([(vw.w2i[sent[i]] if len(sent) > i else STOP) for sent in sents])
        mask = [(1 if len(sent) > i else 0) for sent in sents]
        masks.append(mask)
        tot_words += sum(mask)

    # start the rnn by inputting "<start>"
    init_ids = [START] * len(sents)
    s = f_init.add_input(dy.lookup_batch(WORDS_LOOKUP, init_ids))

    # feed word vectors into the RNN and predict the next word
    losses = []
    for wid, mask in zip(wids, masks):
        # calculate the softmax and loss
        score = W_exp * s.output() + b_exp
        loss = dy.pickneglogsoftmax_batch(score, wid)
        # mask the loss if at least one sentence is shorter
        if mask[-1] != 1:
            mask_expr = dy.inputVector(mask)
            mask_expr = dy.reshape(mask_expr, (1,), MB_SIZE)
            loss = loss * mask_expr
        losses.append(loss)
        # update the state of the RNN
        wemb = dy.lookup_batch(WORDS_LOOKUP, wid)
        s = s.add_input(wemb)

    return dy.sum_batches(dy.esum(losses)), tot_words

num_tagged = cum_loss = 0
# Sort training sentences in descending order and count minibatches
train.sort(key=lambda x: -len(x))
#test.sort(key=lambda x: -len(x))
train_order = [x*MB_SIZE for x in range(len(train)/MB_SIZE)]
#test_order = [x*MB_SIZE for x in range(len(test)/MB_SIZE)]


#print "Train order", train_order

start_time = time.time()

# Perform training
for ITER in xrange(11):
    random.shuffle(train_order)
    print "Shuffled"
    for i, sid in enumerate(train_order, 1):
        print "Batch", i, "of", len(train)/MB_SIZE, "iter", str(ITER)
        loss_exp, mb_words = calc_lm_loss(train[sid:sid+MB_SIZE])
        cum_loss += loss_exp.scalar_value()
        num_tagged += mb_words
        loss_exp.backward()
        trainer.update()
    print "epoch %r finished" % ITER
    model.save("C:\\corpora\\back_batch_bigmodel.txt", [RNN, WORDS_LOOKUP, W_sm, b_sm])
    #Log the line
    log_file = "C:\\corpora\\log.txt"
    logline = "iteration back" + " " + str(ITER) + "\n"
    with open(log_file, "a") as myfile:
        myfile.write(logline)

    trainer.update_epoch(1.0)
'''


(RNN, WORDS_LOOKUP, W_sm, b_sm) = model.load("C:\\corpora\\batch_bigmodel.txt")


def predictNextWord( sentence, builder, wlookup, mR, mB):
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

sentence = "the results clearly"
start_sentence = ['start'] + sentence.split(" ")
#start_sentence = ['<start>', 'the', 'results', 'clearly']
isent = [vw.w2i[w] for w in start_sentence]
print "ISENT", isent
distribution = predictNextWord(isent, RNN, WORDS_LOOKUP, W_sm, b_sm).npvalue()


for i in range(0, 20):
    max_index = np.argmax(distribution)
    print i, vw.w2i.keys()[vw.w2i.values().index(max_index)]
    distribution = np.delete(distribution, max_index)


sentence = "the results clearly"
start_sentence = ['start'] + sentence.split(" ")
#start_sentence = ['<start>', 'the', 'results', 'clearly']
isent = [vw.w2i[w] for w in start_sentence]
print "ISENT", isent
distribution = predictNextWord(isent, RNN, WORDS_LOOKUP, W_sm, b_sm).npvalue()


for i in range(0, 20):
    max_index = np.argmax(distribution)
    print i, vw.w2i.keys()[vw.w2i.values().index(max_index)]
    distribution = np.delete(distribution, max_index)

sentence = "for the collocation extraction"
start_sentence = ['start'] + sentence.split(" ")
#start_sentence = ['<start>', 'the', 'results', 'clearly']
isent = [vw.w2i[w] for w in start_sentence]
print "ISENT", isent
distribution = predictNextWord(isent, RNN, WORDS_LOOKUP, W_sm, b_sm).npvalue()


for i in range(0, 20):
    max_index = np.argmax(distribution)
    print i, vw.w2i.keys()[vw.w2i.values().index(max_index)]
    distribution = np.delete(distribution, max_index)



sentence = "prioritizing these alerts will help security personnel focus their efforts on the"
start_sentence = ['start'] + sentence.split(" ")
#start_sentence = ['<start>', 'the', 'results', 'clearly']
isent = [vw.w2i[w] for w in start_sentence]
print "ISENT", isent
distribution = predictNextWord(isent, RNN, WORDS_LOOKUP, W_sm, b_sm).npvalue()


for i in range(0, 20):
    max_index = np.argmax(distribution)
    print i, vw.w2i.keys()[vw.w2i.values().index(max_index)]
    distribution = np.delete(distribution, max_index)
'''
end_time = time.time()
print "Time:", end_time-start_time