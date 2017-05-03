# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter, defaultdict
from itertools import count
import nltk
import mmap
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


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

        description_file = "/Users/macbook/Desktop/corpora/Yahoo/descr.tsv"

        while data:

            parts = data.split(",")

            description = ""
            qid = parts[0]
            with file(description_file) as f:
                for l in f:
                    description_parts = l.split("\t")
                    if qid == description_parts[0]:
                        description += description_parts[1]
                        #print "added:", description
            end = len(parts)-1
            text_parts = parts[1 : end]
            line = ",".join(text_parts)
            data = m.readline()
            line = line.lower() + description.lower()
            line = ExtractAlphanumeric(line)
            tokens = nltk.word_tokenize(line)
            line = ["<start>"] + tokens + ["<stop>"]
            print len(line)
            #print line
            yield line


def readY(fname):
    Ys = []
    with file(fname) as fh:
        for line in fh:
            line = line.lower()
            Ys.append(int(line.strip()[-1]))
    return Ys

filename = "/Users/macbook/Desktop/corpora/Yahoo/Title_3.csv"

train = FastCorpusReaderYahoo(filename)

vocab = Vocab.from_corpus(train)

WORDS_NUM = vocab.size()
#print "NUM of WORDS", WORDS_NUM


Ys = readY(filename)
train = list(train)

complete_text = ""


help_vocab = dict()
for sent in train:
    sent_len = len(sent)
    current_sent = " ".join(sent)
    complete_text += current_sent + " "
    if sent_len in help_vocab.keys():
        help_vocab[sent_len] += 1
    else:
        help_vocab[sent_len] = 1


fdist = nltk.FreqDist(complete_text)

number_of_words = 5000
most_common_words = fdist.most_common(number_of_words)


def is_common(common_list, word):
    for pair in common_list:
        if pair[0] == word:
            return True
    return False


int_train = []
i = 0
max_sent_length = 100

for sentence in train:
    isent = [vocab.w2i[w] for w in sentence]
    int_train.append(isent)

#print train

print len(int_train)
print len(Ys)


recall_1_list = []
recall_0_list = []
auc = []

# fix random seed for reproducibility
numpy.random.seed(7)

kf = model_selection.KFold(n_splits=5)
for train_idx, test_idx in kf.split(int_train):

    X_train = [int_train[i] for i in train_idx]
    Y_train = [Ys[i] for i in train_idx]

    X_test = [int_train[i] for i in test_idx]
    Y_test = [Ys[i] for i in test_idx]

    X_train = sequence.pad_sequences(X_train, maxlen=max_sent_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sent_length)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(WORDS_NUM, embedding_vecor_length, input_length=max_sent_length))
    model.add(Dropout(0.2))
    model.add(LSTM(300))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train, nb_epoch=4, batch_size=64)

    predictions = model.predict(X_test)

    auc.append(roc_auc_score(Y_test, predictions))

    rounded = []
    for pred in predictions:
        if pred >0.5:
            rounded.append(1)
        else:
            rounded.append(0)

    recall_0_list.append(recall_score(Y_test, rounded, pos_label=0))
    recall_1_list.append(recall_score(Y_test, rounded, pos_label=1))



print "RECALL 0:", sum(recall_0_list) / float(len(recall_0_list))
print "RECALL 1:", sum(recall_1_list) / float(len(recall_1_list))
print "AUC :", sum(auc)/float(len(auc))