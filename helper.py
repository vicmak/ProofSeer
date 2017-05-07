
def get_conll():
    filename = "/Users/macbook/Desktop/corpora/official-2014.combined.m2"
    good_counter = 0
    bad_counter = 0
    with open(filename) as f:
        old_index = -1
        current_sent = "fuck"
        old_sent = []
        for line in f:
            tokens = line.split(" ")
            if len(tokens) > 1:
                if tokens[0] == "S":
                    #print line
                    current_sent = tokens
                if tokens[0] == "A":
                    ann_tokens = line.split("|||")
                    if ann_tokens[1] == "Wtone":
                        offsets = ann_tokens[0].split(" ")
                        start_off = int(offsets[1]) + 1
                        if (start_off != old_index):
                            end_off = int(offsets[2])
                            old_sent = list(current_sent)
                            most_current_sent = list(current_sent)
                            most_current_sent[start_off] = "*" + most_current_sent[start_off] + "/" + ann_tokens[2].strip()
                            #print most_current_sent
                            most_current_sent[-1]="."
                            if len(ann_tokens[2].split(" ")) == 1 and ann_tokens[2].isalpha():
                                print " ".join(most_current_sent[1:])
                                good_counter+=1
                            else:
                                bad_counter+=1
                            current_sent[start_off] = ann_tokens[2]
                            old_index = start_off
                        else:
                            old_sent[start_off] = "*" + old_sent[start_off] + "/" + ann_tokens[2].strip()
                            old_sent[-1] = "."
                            if len(ann_tokens[2].split(" ")) == 1 :
                                print " ".join(old_sent[1:])
                                good_counter+=1
                            else:
                                bad_counter +=1
    print good_counter, bad_counter

def main():

    #get_conll()
    '''
    filename = "/Users/macbook/Desktop/corpora/conll_one.txt"
    with open(filename) as f:
        for line in f:
            tokens = line.split(" ")
            for token in tokens:
                if len(token) > 0:
                    if token[0] == "*":
                        if token.endswith("/ "):
                            print token
    '''
if __name__ == "__main__":
    main()