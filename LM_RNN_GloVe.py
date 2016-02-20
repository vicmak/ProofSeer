from sklearn import preprocessing
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
import wvlib
import random
from random import randint
import matplotlib.pyplot as plt
from scipy import spatial
import linecache

def add_unseen_token_2_extra_vocabulary(token, extra_vocab_filename):
    print("Adding unseen token: ", token)
    random_vector = [random.random() for _ in range(0, 300)]
    string_vector = [str(i) for i in random_vector]
    vector = [token.lower()] + [" "] + string_vector
    with open(extra_vocab_filename, "a") as myfile:
        str_vector = ' '.join(str(e) for e in vector) #covert list to string
        str_vector = str_vector + "\n"
        myfile.write(str_vector)
    return random_vector

def tokenize_file_to_vectors(vectors_file, file2tokenize, outputfile):
    with open(file2tokenize) as f:
        for line in f:
            tokens = line.split()
            for token in tokens:
                print("tokenizing: ",token)
                token = my_strip(token)
                vector = get_vector(token, vectors_file)
                #vector = get_test_vector()
                with open(outputfile, "a") as myfile:
                    str_vector = ','.join(str(e) for e in vector) #covert list to string
                    str_vector = str_vector + "\n"
                    myfile.write(str_vector)

def my_strip (token):
    token = token.strip(",")
    token = token.strip(".")
    token = token.strip("?")
    token = token.strip(":")
    return token

def get_test_vector():
    return list(range(10,15))


def get_vector(token, vectors_file, extra_vectors_file="/Users/macbook/Desktop/corpora/extra_vocab.txt"):

    with open(extra_vectors_file) as f:
        for line in f:
            tokens = line.split()
            if tokens[0] == token.lower():
                vec = tokens[1:301]
                print ("returning from extra: ", token)
                return vec

    with open(vectors_file) as f:
        for line in f:
            tokens = line.split()
            if tokens[0] == token.lower():
                vec = tokens[1:301]
                return vec

    vec = add_unseen_token_2_extra_vocabulary(token,extra_vectors_file)
    return vec

def get_vector_by_number(token_number, vectors_filename="/Users/macbook/Desktop/corpora/glove.42B.300d.txt"):
    line = linecache.getline(vectors_filename,token_number)
    tokens = line.split()
    vec = tokens[1:301]
    return [float(i) for i in vec]

def load_data(train_file_name, window_size=10):

    #train = pd.read_csv(train_file_name, header=None, delim_whitespace=True)
    train = pd.read_csv(train_file_name)
    docX, docY = [], []

    for i in range(len(train)-window_size):
        docX.append(train.iloc[i:i+window_size].as_matrix())
        docY.append(train.iloc[i+window_size-1].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    print("Shape x:", alsX.shape)
    print("Shape y:", alsY.shape)


    return alsX, alsY

def run_experiment(tokenized_by_vectors_filename):

    X, Y = load_data(tokenized_by_vectors_filename)

    in_out_neurons = 300
    out_n = 300
    hidden_neurons = 30

    #create the model here
    print("Creating model...")
    model = Sequential()
    print("Adding LSTM ...")
    model.add(GRU(hidden_neurons, input_dim=in_out_neurons, return_sequences=False))

    print("Adding dropout ...")
    model.add(Dropout(0.8))
    print("adding output layer...")
    model.add(Dense(out_n, input_dim=hidden_neurons))
    print("adding activation...")
    model.add(Activation("linear"))
    print("compiling...")
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    print("compiled!")
    #split the data to train and test

    data_size = X.shape[0]
    train_size = int(data_size * 0.7)

    X_train = X[0:train_size, :]
    Y_train = Y[0:train_size]
    Y_test = Y[train_size+1:data_size]
    X_test = X[train_size+1:data_size, :]

    print("start train!")

    model.fit(X_train, Y_train, batch_size=100, nb_epoch=300, validation_split=0.05)

    print ("model is fit!")

    predicted_results = model.predict(X_test)


    #print ("randome predicted shape: ", random_results.shape)
    random_results = []
    for i in range(1,predicted_results.shape[0]):
        random_results.append(get_vector_by_number(randint(1,10000)))

    calc_error_distribution(Y_test,predicted_results,random_results)



def calc_error_distribution(correct_values, predicted_values, random_values):

    test_errors_file_name = "/Users/macbook/Desktop/corpora/test_errors.txt"
    test_random_errors_file_name = "/Users/macbook/Desktop/corpora/test_random_errors.txt"

    predicted_dist = []
    random_dist = []

    with open(test_errors_file_name, "a") as myfile:
        for i in range(0,len(correct_values)-1):
            distance = calc_distance(correct_values[i], predicted_values[i])
            predicted_dist.append(distance)
         #   myfile.write(" " + str(distance))

    with open(test_random_errors_file_name, "a") as myfile:
        for i in range(0,len(correct_values)-1):
            distance = calc_distance(correct_values[i], random_values[i])
            random_dist.append(distance)
          #  myfile.write(" " + str(distance))

    plt.hist(predicted_dist, fc=(0, 0, 1, 0.5), label="predicted")
    plt.hist(random_dist, fc=(1, 0, 0, 0.5), label="random")
    plt.legend(loc='upper right')
    plt.show()

def calc_distance(vec1, vec2):
    return spatial.distance.cosine(vec1, vec2)

def main():



    file_2_tokenize_name = "/Users/macbook/Desktop/corpora/text2tokenize.txt"
    file_2_tokenize_name_test = "/Users/macbook/Desktop/corpora/text2tokenize_test.txt"
    tokenized_file_name = "/Users/macbook/Desktop/corpora/tokenized2vectors.txt"
    glove_vectors_file_name = "/Users/macbook/Desktop/corpora/glove.42B.300d.txt"
    vectors_test_file_name = "/Users/macbook/Desktop/corpora/vectors_test.txt"
    extra_vocab_filename = "/Users/macbook/Desktop/corpora/extra_vocab.txt"
    train_file_name = "/Users/macbook/Desktop/corpora/tokenized_train.txt"
    test_errors_file_name = "/Users/macbook/Desktop/corpora/test_errors.txt"
    test_random_errors_file_name = "/Users/macbook/Desktop/corpora/test_random_errors.txt"

    tokenize_file_to_vectors(glove_vectors_file_name, file_2_tokenize_name, tokenized_file_name)
    print("FINISHED TOKENIZING")
    run_experiment(tokenized_file_name)


   # X, Y = load_data(tokenized_file_name)
    #print ("haha")
    #wv = wvlib.load("/Users/macbook/Desktop/corpora/glove.42B.300d.txt")
    #print("loaded")
    #print("cat dog sim:", wv.similarity("dog", "cat"))
    #print("nearest to dog", wv.nearest("dog"))

   # run_experiment()
    #load_data(train_file_name)




if __name__ == "__main__":
    main()