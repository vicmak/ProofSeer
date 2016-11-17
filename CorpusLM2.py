import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import os
import time
import random
from random import randint

from cStringIO import StringIO


class GloveConfig(object):
    vector_dimension = 300
    epochs_number = 50
    glove_vectors = "C:\\corpora\\MSCC\\vectors_glove_mscc_300d_nostopwords.txt"


def add_unseen_token_2_extra_vocabulary(token, extra_vocab_filename):
    config = GloveConfig()
    print("Adding unseen token: ", token)
    random_vector = [random.random() for _ in range(0, config.vector_dimension)]
    string_vector = [str(i) for i in random_vector]
    vector = [token.lower()] + [" "] + string_vector
    with open(extra_vocab_filename, "a") as myfile:
        str_vector = ' '.join(str(e) for e in vector)  # covert list to string
        str_vector = str_vector + "\n"
        myfile.write(str_vector)
    return ','.join(string_vector)


def get_vector_string(token, vectors_dict,
                      extra_vectors_file="C:\\corpora\\MSCC\\extra_vocab.txt"):
    config = GloveConfig()

    if token in vectors_dict.keys():
        return vectors_dict[token]

    with open(extra_vectors_file) as f:
        for line in f:
            tokens = line.split()
            if tokens[0] == token.lower():
                vec = tokens[1:config.vector_dimension + 1]
                print ("returning from extra: ", token)
                return ','.join(vec)



    vec = add_unseen_token_2_extra_vocabulary(token, extra_vectors_file)
    return vec


def read_vocab_to_list(filename):
    return [word for line in open(filename, 'r') for word in line.split()]


def read_vectors_to_dict(vectors_filename):
    config = GloveConfig()
    vectors_dict = dict()
    with open(vectors_filename) as f:
        for line in f:
            tokens = line.split()
            vec = tokens[1:config.vector_dimension + 1]
            vec_string = ",".join(vec)
            vectors_dict[tokens[0]] = vec_string
    return vectors_dict


def clean_text(token):
    token = token.replace(",", "")
    token = token.replace(".", "")
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


def find_word_index(word, common_words_filename):
    count = 0
    if len(word) > 0:
        with open(common_words_filename) as f:
            for line in f:
                tokens = line.split()
                # print(str(len(tokens)))
                if tokens[0] == word:
                    return count
                count = count + 1
    else:
        print("Empty token")
    return -1


def find_ngrams(s, n):
    input_list = s.split(" ")
    for i in range(0, len(input_list), 1):
        input_list[i] = clean_text(input_list[i])
    return zip(*[input_list[i:] for i in range(n)])


def log_train_file(filename, tokens_num):
    log_file = "C:\\corpora\\MSCC\\log_mscc.txt"
    logline = filename + " " + str(tokens_num) + "\n"
    with open(filename, "a") as myfile:
        myfile.write(logline)


def get_tokenized_file_to_vectors(vocab, file2tokenize, vectors, context_size=4, target_word_index=2):
    count=0
    tokenized_file = ""
    with open(file2tokenize) as f:
        for line in f:
            line = line.lower()
            tokens = line.split()
            if len(tokens) > context_size:
                n_grams = find_ngrams(line, context_size + 1)
                print ("ngrams amount:", len(n_grams))
                for gram in n_grams:
                    current_y = gram[target_word_index]
                    if current_y in vocab:
                        count = count + 1
                        print(count, len(n_grams), gram)
                        str_gramm = get_csv(gram, vocab, vectors, target_word_index, context_size + 1)
                        tokenized_file = tokenized_file + str_gramm
    return tokenized_file


def get_csv(gram, vocab, vectors, target_index,
            n_gram_size):  # target index is the middle word in window. context-leaf and context-right are same length.

    xs = get_xs(gram, target_index, vectors, n_gram_size)
    ys = get_ys(gram, target_index, vocab)
    result = xs + ys + "\n"
    return result


def get_xs(gram, exclude_index, vectors, n_gram_size):
    xs = ""
    for i in range(0, n_gram_size, 1):
        if i != exclude_index:
            xs = xs + get_vector_string(gram[i], vectors) + ","
    return xs


def get_ys(gram, label_index, vocab):
    vocab_size = len(vocab)
    ys = ["0"] * vocab_size
    index = find_word_index_in_list(gram[label_index], vocab)
    if index != -1:
        ys[index] = "1"
    return ','.join(str(e) for e in ys)


def find_word_index_in_list(word, word_list):
    if word in word_list:
        return word_list.index(word)
    return -1


def train_model_from_dir(root, vocabulary, vectors):
    word_dimension = 300
    context_size = 4
    hidden_layer_size = 100
    vocabulary_size = 30000
    log_file_name = "C:\\corpora\\MSCC\\log_mscc.txt"
    model_save_file_name = "C:\\corpora\\models\\model.h5"

    start_time = time.time()
    print("Creating the model object")
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=context_size * word_dimension, init='uniform', activation='tanh'))
    model.add(Dense(vocabulary_size, init='normal', activation='softmax'))  # can be also sigmoid (for a multiclass)
    print("compiling...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("compiled!")

    count = 0
    for path, subdirs, files in os.walk(root):
        for name in files:
            current_filename = os.path.join(path, name)
            if current_filename.endswith("TXT"):
                if find_word_index(current_filename, log_file_name) == -1:  # the file is not logged
                    count = count + 1
                    print("file number", count)
                    print current_filename
                    data = get_tokenized_file_to_vectors(vocabulary, current_filename, vectors)
                    print("got data")
                    dataset = np.genfromtxt(StringIO(data), delimiter=",")
                    print("got dataset")
                    X = dataset[:, 0:word_dimension * context_size]
                    Y = dataset[:, word_dimension * context_size:]
                    arrX = np.array(X)
                    arrY = np.array(Y)
                    model.fit(arrX, arrY, nb_epoch=50, batch_size=dataset.shape[0])  # check the batch size
                    log_train_file(current_filename, dataset.shape[0])
                    if count % 10 == 0:
                        print("Saving model...", count)
                        model.save(model_save_file_name)
                else:
                    print("file already trained:", current_filename)

    end_time = time.time()
    print("elapsed time", end_time - start_time)
    model.save(model_save_file_name)


def train_model_from_dir_batch(root, vocabulary, vectors):
    word_dimension = 300
    context_size = 4
    hidden_layer_size = 100
    vocabulary_size = 30000
    log_file_name = "C:\\corpora\\MSCC\\log_mscc.txt"
    model_save_file_name = "C:\\corpora\\models\\model.h5"

    start_time = time.time()
    print("Creating the model object")
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=context_size * word_dimension, init='uniform', activation='tanh'))
    model.add(Dense(vocabulary_size, init='normal', activation='softmax'))  # can be also sigmoid (for a multiclass)
    print("compiling...")
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("compiled!")

    count = 0
    for path, subdirs, files in os.walk(root):
        for name in files:
            current_filename = os.path.join(path, name)
            if current_filename.endswith("TXT"):
                if find_word_index(current_filename, log_file_name) == -1:  # the file is not logged
                    count = count + 1
                    print("file number", count)
                    print current_filename
                    model.fit_generator(n_gram_generator(vocabulary,current_filename, vectors), samples_per_epoch=100,
                                        nb_epoch=50)
                    log_train_file(current_filename, 666)
                    print("Saving model...", count)
                    model.save(model_save_file_name)
                else:
                    print("file already trained:", current_filename)

    end_time = time.time()
    print("elapsed time", end_time - start_time)
    model.save(model_save_file_name)


def n_gram_generator(vocab, file2tokenize, vectors, context_size=4, target_word_index=2):
    config = GloveConfig()
    count = 0
    with open(file2tokenize) as f:
        for line in f:
            line = line.lower()
            tokens = line.split()
            if len(tokens) > context_size:
                n_grams = find_ngrams(line, context_size + 1)
                print ("ngrams amount:", len(n_grams))
                for gram in n_grams:
                    current_y = gram[target_word_index]
                    if current_y in vocab:
                        count = count + 1
                        print(count, len(n_grams), gram)
                        str_gram = get_csv(gram, vocab, vectors, target_word_index, context_size + 1)
                        dataset = np.genfromtxt(StringIO(str_gram), delimiter=",")
                        X = dataset[0:config.vector_dimension * context_size]
                        Y = dataset[config.vector_dimension * context_size:]

                        arrX = np.array(X)
                        arrY = np.array(Y)
                        print arrX.shape
                        print arrY.shape
                      #  print("just before yield")
                        yield arrX, arrY


def main():
    stop_words_filename = "C:\\corpora\\MSCC\\mscc_stop_words.txt"
    dense_vectors_glove = "C:\\corpora\\MSCC\\vectors_glove_mscc_300d_nostopwords.txt"
    vocabulary_filename = "C:\\corpora\\MSCC\\mscc_clean_vocab_30000.txt"
    vocab = read_vocab_to_list(vocabulary_filename)
    vectors = read_vectors_to_dict(dense_vectors_glove)
    print(vectors['can'])

    print len(vectors)
    train_model_from_dir_batch("C:\\corpora\\mscc_small", vocab, vectors)

    #  continue_train_model_from_dir("/Users/macbook/Desktop/corpora/corpus30k",
    #                                vocab,
    #                                "/Users/macbook/Desktop/corpora/aux_files/model.h5")

    #  print("Starting first model")
    #  test_model_on_dir("/Users/macbook/Desktop/corpora/aux_files/model115.h5", "/Users/macbook/Desktop/corpora/triple_test", vocab)
    # test_model_on_dir("/Users/macbook/Desktop/corpora/aux_files/model2500.h5", "/Users/macbook/Desktop/corpora/triple_test", vocab)
    #  print(data)
    # print (x)


if __name__ == "__main__":
    main()