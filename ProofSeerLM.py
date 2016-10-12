from sklearn import preprocessing
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.models import load_model
from keras.layers.recurrent import LSTM, GRU
import wvlib
import random
from random import randint
import matplotlib.pyplot as plt
from scipy import spatial
import linecache
import os
import time

from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


def predictWordByContext(context):
    return 1


def my_strip(token):
    token = token.strip(",")
    token = token.strip(".")
    token = token.strip("?")
    token = token.strip(":")
    return token


def tokenize_file_to_vectors(common_words_file, file2tokenize, outputfile):
    with open(file2tokenize) as f:
        for line in f:
            tokens = line.split()
            if (len(tokens) > 4):
                five_gramms = find_ngrams(line, 5)
                for gramm in five_gramms:
                    str_gramm = get_csv(gramm)
                    with open(outputfile, "a") as myfile:
                        myfile.write(str_gramm)


def get_tokenized_file_to_vectors(common_words_file, file2tokenize):
    tokenized_file = ""
    with open(file2tokenize) as f:
        for line in f:
            tokens = line.split()
            if len(tokens) > 4:
                five_gramms = find_ngrams(line, 5)
                for gramm in five_gramms:
                    str_gramm = get_csv(gramm)
                    tokenized_file = tokenized_file + str_gramm
    return tokenized_file


def get_csv(gramm):
    xs = get_xs(gramm, 2)
    ys = get_ys(gramm, 2)
    result = xs + " , " + ys + "\n"
    return result


def find_word_index(word):
    common_words_filename = "/Users/macbook/Desktop/corpora/aux_files/magic.txt"
    count = 0;
    with open(common_words_filename) as f:
        for line in f:
            tokens = line.split()
           # print(str(len(tokens)))
            if tokens[0] == word:
                return count
            count = count +1
    #return -1 if the word was not found in the list of most common words
    return -1


def write_common_files():
    with open("/Users/macbook/Desktop/corpora/magic.txt", "a") as myfile:
         with open("/Users/macbook/Desktop/corpora/common_words_coca.txt") as f:
            for line in f:
                tokens = line.split()
                for i in range(0, 4999, 1):
                    print("writing: " + tokens[i][3:])
                    myfile.write(tokens[i][3:] + "\n")


def get_xs(gramm, exclude_index):

    xs = ["0"] * 10000
    for i in range(0, 4, 1):
        if i != exclude_index:
            index = find_word_index(gramm[i])
            if (index != -1):
                xs[index] = "1"
              # print(index)
    return ','.join(str(e) for e in xs)


def get_ys(gramm, label_index):
    ys = ["0"] * 10000
    index = find_word_index(gramm[label_index])
    if (index != -1):
        ys[index] = "1"
    return ','.join(str(e) for e in ys)


def find_ngrams(s, n):
    input_list = s.split(" ")
    return zip(*[input_list[i:] for i in range(n)])


def create_online_dataset(filename):

    np.genfromtxt(StringIO(data), delimiter=",")

def getModel(one_hot_train_filename):

    print("Reading Dataset from file")
    dataset = np.loadtxt(one_hot_train_filename, delimiter=",")
    X = dataset[:, 0:10000]
    Y = dataset[:, 10000:]

    arrX = np.array(X)
    arrY = np.array(Y)

    print("Dataset read!")
    print(arrX.shape)
    print(arrY.shape)

    # Create the model
    print("Creating the model object")
    model = Sequential()
    model.add(Dense(5000, input_dim=10000, init='uniform', activation='relu'))
    model.add(Dense(10000, init='normal', activation='softmax'))  # can be also sigmoid (for a multiclass)

    # Compile the model
    print("compiling...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("compiled!")

    # Train the model
    print("Start training the model")

    print("fitted")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    model.fit(X_train, Y_train, nb_epoch=3, batch_size=100)
    predictions = model.predict(X_test)

    print("predictions")
    print(predictions)

    for i in range(0,len(predictions),1):
        print (np.argmax(predictions[i]))


def train_model_from_dir(root, vocabulary_filename):

    print("Creating the model object")
    model = Sequential()
    model.add(Dense(500, input_dim=10000, init='uniform', activation='relu'))
    model.add(Dense(10000, init='normal', activation='softmax'))  # can be also sigmoid (for a multiclass)
    print("compiling...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("compiled!")
    count = 0
    for path, subdirs, files in os.walk(root):
        for name in files:
            count = count + 1
            print("file number", count)
            data = get_tokenized_file_to_vectors(vocabulary_filename, os.path.join(path, name))
            dataset = np.genfromtxt(StringIO(data), delimiter=",")
            X = dataset[:, 0:10000]
            Y = dataset[:, 10000:]
            arrX = np.array(X)
            arrY = np.array(Y)
            model.fit(arrX, arrY, nb_epoch=3, batch_size=100)

    model.save("/Users/macbook/Desktop/corpora/aux_files/model.h5")


def test_model_on_dir(model_filename, root, vocabulary_filename):
    count = 0
    model = load_model(model_filename)
    print("model loaded")
    for path, subdirs, files in os.walk(root):
        for name in files:

            print("file number", os.path.join(path, name))
            data = get_tokenized_file_to_vectors(vocabulary_filename, os.path.join(path, name))
            dataset = np.genfromtxt(StringIO(data), delimiter=",")
            print("loaded dataset")
            X = dataset[:, 0:10000]
            Y = dataset[:, 10000:]
            arrX = np.array(X)
            arrY = np.array(Y)
            predictions = model.predict(arrX)
            mrr = getMRR(predictions, arrY)
            print("MRR:", mrr)


def getMRR(predictions, labels):
    count = 0
    sum = 0
    for i in range(0,len(predictions), 1):
        correct_index = np.argmax(labels[i])
        predicted_index = np.argmax(predictions[i])
     #   print("correct and predicted:", correct_index, predicted_index)
        rank = 1
        count = 0
        while (predicted_index != correct_index and rank<len(predictions)):
            predictions[predicted_index]= -1
            predicted_index = np.argmax(predictions[i])
         #   print("predicted index:", predicted_index)
            rank = rank + 1
        if rank < len(predictions)-1:
            sum = sum + 1/float(rank)
    return sum/float(len(predictions))


def main():

    print("haha")

    common_words_filename = "/Users/macbook/Desktop/corpora/aux_files/magic.txt"
    text_from_pdf_file = "/Users/macbook/Desktop/corpora/aux_files/text_from_pdf_dir.txt"
    file_2_tokenize_name = "/Users/macbook/Desktop/corpora/aux_files/qpp1.txt"
    one_hot_train_data = "/Users/macbook/Desktop/corpora/aux_files/one_hot_csv.txt"
    test_model_on_dir("/Users/macbook/Desktop/corpora/aux_files/model.h5","/Users/macbook/Desktop/corpora/tiny_test", common_words_filename)
    train_model_from_dir("/Users/macbook/Desktop/corpora/tiny_test", common_words_filename)

   # write_common_files()
    #tokenize_file_to_vectors(common_words_filename, file_2_tokenize_name, one_hot_train_data)
  #  x = predictWordByContext("bla")
  #  getModel(one_hot_train_data)
   # data = get_tokenized_file_to_vectors(common_words_filename, file_2_tokenize_name)
    #dataset = np.genfromtxt(StringIO(data), delimiter=",")

    #X = dataset[:, 0:10000]
    #Y = dataset[:, 10000:]

    #arrX = np.array(X)
    #arrY = np.array(Y)

    #print("Dataset read!")
    #print(arrX.shape)
    #print(arrY.shape)


  #  print(data)
   # print (x)


if __name__ == "__main__":
    main()
