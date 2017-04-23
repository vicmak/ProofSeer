from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn import svm

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


def read(fname):

    train = []

    with file(fname) as fh:
        for line in fh:
            line = line.lower()
            sent = line.strip().split(",")[1]
            sent = clean_text(sent)
            sent = "<start> " + sent + " <stop>"
            train.append(sent)
    return train


def readY(fname):
    train = []
    with file(fname) as fh:
        for line in fh:
            line = line.lower()
            sent = line.strip().split(",")[2]
            sent = clean_text(sent)
            #print sent
            train.append(int(sent))
    return train

def main():

    datafile = "/Users/macbook/Desktop/corpora/Yahoo/Title_1.csv"
    train = read(datafile)
    vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)

    Y = readY(datafile)
    kf = model_selection.KFold(n_splits=5)


    X = vectorizer.fit_transform(train).toarray()

    clf = svm.SVC(kernel='linear', C=1)
   # scores = cross_val_score(clf, X, Y, cv=5)

    #print scores

    for train_idx, test_idx in kf.split(train):

        X_train =  [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]

        Y_train = [Y[i] for i in train_idx]
        Y_test = [Y[i] for i in test_idx]

        clf.fit(X_train,Y_train)
        res = clf.predict(X_test)

        correct = 0

        for i in range(0,len(res)):
            if res[i] == Y_test[i]:
                correct +=1

        print correct / float(len(res))


if __name__ == "__main__":
    main()