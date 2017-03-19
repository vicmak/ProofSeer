import os
import operator
import io
import gensim

import os
import operator
import io

def my_strip(token):
    token = token.strip(",")
    token = token.strip(".")
    token = token.strip("?")
    token = token.strip(":")
    return token


def clean_text(token):
    token = token.replace(",", "")
    #token = token.replace(".", "")
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


def get_clean_text_from_file(filename, stop_words_list=None):

    file_content = ""
    with open(filename) as f:
        for line in f:
            line = line.lower()
            tokens = line.split()
            for token in tokens:
                clean_token = clean_text(token)
                if len(clean_token) > 1 and clean_token.isalpha():
                    if stop_words_list != None:
                        if clean_token not in stop_words_list:
                            file_content = file_content + clean_token + " "
                    else:
                        file_content = file_content + clean_token + " "
    return file_content


def get_clean_text_from_file2(filename):

    file_content = ""
    count = 0
    with open(filename) as f:
        for line in f:
            count = count + 1
            if count > 245:
                file_content = file_content + line.lower() + " "
    return file_content

def get_file_properties(filename):

    counts = dict()
    file_content = ""
    count = 0
    with open(filename) as f:
        for line in f:
            line_length = len(line.lower().split())
            if counts.has_key(line_length):
                counts[line_length] = counts[line_length] + 1
            else:
                counts[line_length] = 1

    for i in range(41, 257):
        if counts.has_key(i):
            count = count + counts[i]
            print i
    print "count", counts
    print "count", count

def read_vocab_to_list(filename):
    return [word for line in open(filename, 'r') for word in line.split()]


def read_vocab(filename):
    vocab = []
    with open(filename, "r") as myfile:
        text = myfile.read()
        rows = text.split("\n")
        print rows
        for row in rows:
            vocab.append(row.split(" ")[0])
    return vocab

def clean_mscc(source_dir, target_dir):
    count = 0
    for root, dirs, files in os.walk(source_dir):
        path = root.split('/')
        print (len(path) - 1) *'---' , os.path.basename(root)
        for file in files:
            if file.endswith("TXT"):
                count = count + 1
                print("count: ", count)
                current_source_file_name = source_dir + "/" + file
                target_file_text = get_clean_text_from_file2(current_source_file_name)
                current_target_file_name = target_dir + "/" + file
                with io.FileIO(current_target_file_name, "w") as file_target:
                    file_target.write(target_file_text)


def create_single_file(source_dir, target_filename):
    count = 0
    for root, dirs, files in os.walk(source_dir):
        path = root.split('/')
        print (len(path) - 1) *'---' , os.path.basename(root)
        for file in files:
            if file.endswith("TXT"):
                count = count + 1
                print("count: ", count)
                current_source_file_name = source_dir + "/" + file
                target_file_text = get_clean_text_from_file(current_source_file_name)
                with io.FileIO(target_filename, "a") as file_target:
                    file_target.write(target_file_text)


def clean_corpus(source_dir, target_dir, stop_words_list):
    count = 0
    for root, dirs, files in os.walk(source_dir):
        path = root.split('/')
        print (len(path) - 1) *'---' , os.path.basename(root)
        for file in files:
            if file.endswith("TXT"):
                count = count + 1
                print("count: ", count)
                current_source_file_name = source_dir + "/" + file
                target_file_text = get_clean_text_from_file(current_source_file_name, stop_words_list)
                current_target_file_name = target_dir + "/" + file
                with io.FileIO(current_target_file_name, "w") as file_target:
                    file_target.write(target_file_text)


def extract_vocabulary(dir_name, targetTextFileName="bla"):
    count = 0
    vocabulary = dict()
    for root, dirs, files in os.walk(dir_name):
        path = root.split('/')
        print (len(path) - 1) *'---' , os.path.basename(root)
        for file in files:
            if (file.endswith("txt")):
                count = count + 1
                print("count: ", count)
                with open(dir_name + "/" + file) as f:
                    for line in f:
                        line = line.lower()
                        tokens = line.split()
                        for token in tokens:
                            clean_token = clean_text(token)
                            if vocabulary.has_key(clean_token):
                                vocabulary[clean_token] = vocabulary[clean_token] + 1
                            else:
                                vocabulary[clean_token] = 1
    sorted_vocabulary = sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_vocabulary


def flush_txts_to_file(sourceTxtFolderName, targetTextFileName="bla", vocab = None):
    for root, dirs, files in os.walk(sourceTxtFolderName):
        path = root.split('/')
        print (len(path) - 1) *'---' , os.path.basename(root)
        for file in files:
           # print len(path)*'---', file
            if (file.endswith("txt")):
                current_filename = root + "/" + file
                with open(current_filename, "r") as myfile:
                    current_text = myfile.read().lower()
                current_text = clean_text(current_text)
                print ("adding file:", file)
                sentences = current_text.split(".")
                with open(targetTextFileName, "a") as myfile:
                    for i in range(0, len(sentences)):
                        sentence = sentences[i].replace("\n", " ").replace("- ", "")
                       #print sentence
                        if check_vocab_sentence(sentence, vocab):
                            if len(sentence.strip().split(" "))>2:
                                myfile.write("\n" + sentence)


def check_vocab_sentence(sentence, vocab):
    tokens = sentence.split(" ")
    #print tokens
    for token in tokens:
        if token not in vocab:
            return False
    return True


def write_vocab_2_file(filename, vocab):
    vocab_size = 30000
    count = 0
    with open(filename, "w") as myfile:
        for pair in vocab:
            if pair[0].isalpha() and count < vocab_size:
                myfile.write(pair[0] + " " + str(pair[1]) + "\n")
                count += 1

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


def main():

    #sentences = MySentences('C:\\corpora\\mscc_clean')  # a memory-friendly iterator
    #model = gensim.models.Word2Vec(sentences, min_count=27, size=300, workers=4, negative=10, iter=50)
    #model.save('C:\\corpora\\sgns_mscc')
    #model.save_word2vec_format('C:\\corpora\\sgns_mscc_300d.txt','C:\\corpora\\vocab_mscc.txt')

    #clean_corpus("C:\\corpora\\triple_test", "C:\\corpora\\triple_test_clean", stop_words)
    #print("vocab tools")
    #vocab = extract_vocabulary("C:\\corpora\\corpus30k")
    #print(vocab)
    #print(len(vocab))
    #common_words_filename = "C:\\corpora\\vocab.txt"
    #write_vocab_2_file(common_words_filename, vocab)
    #vocab = read_vocab(common_words_filename)
    #print vocab
    #flush_txts_to_file("C:\\corpora\\corpus30k", "C:\\corpora\\long30k.txt",vocab)
    get_file_properties("C:\\corpora\\long30k.txt")
if __name__ == "__main__":
    main()