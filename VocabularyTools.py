import os
import operator

def my_strip(token):
    token = token.strip(",")
    token = token.strip(".")
    token = token.strip("?")
    token = token.strip(":")
    return token


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


def write_vocab_2_file(filename, vocab):
    with open(filename, "w") as myfile:
        for pair in vocab:
            myfile.write(pair[0] + " " + str(pair[1]) + "\n")


def main():

    print("vocab tools")

    vocab = extract_vocabulary("/Users/macbook/Desktop/corpora/corpus30k")

    print(vocab)
    print(len(vocab))
    common_words_filename = "/Users/macbook/Desktop/corpora/aux_files/vocab.txt"

    write_vocab_2_file(common_words_filename, vocab)


if __name__ == "__main__":
    main()