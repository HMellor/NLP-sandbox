import nltk
import string
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer

# load doc into memory
def load_doc(file_path):
    # open the file as read only
    with open(file_path, "r") as f:
        # read all text
        text = f.read()
    return text


# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans("", "", string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# load doc and add to vocab
def add_doc_to_vocab(file_path, vocab):
    # load doc
    doc = load_doc(file_path)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)


# load doc, clean and return line of tokens
def doc_to_line(file_path, vocab):
    # load the doc
    doc = load_doc(file_path)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return " ".join(tokens)


# load all docs in a directory
def process_docs(directory, vocab, is_train=False, generate_vocab=False):
    lines = []
    # walk through all files in the folder
    split = "training" if is_train else "testing"
    desc = "Processing {} data".format(split)
    # list of file paths for this split, negative then positive
    file_paths = []
    for sentiment in ["neg", "pos"]:
        sent_dir = directory / sentiment
        file_paths += [
            f for f in sent_dir.iterdir() if f.name.startswith("cv9") != is_train
        ]
    for file_path in tqdm(file_paths, desc=desc):
        if generate_vocab:
            # add doc to vocab
            add_doc_to_vocab(file_path, vocab)
        else:
            # load and clean the doc
            line = doc_to_line(file_path, vocab)
            # add to list
            lines.append(line)
    return lines


# save list to file
def save_list(lines, file_path):
    # convert lines to a single blob of text
    data = "\n".join(lines)
    # open file
    with open(file_path, "w") as f:
        # write text
        f.write(data)


# prepare bag of words encoding of docs
def prepare_data(train_docs, test_docs, mode):
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the tocuments
    tokenizer.fit_on_texts(train_docs)
    # encode training data set
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    # encode test data set
    Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
    return Xtrain, Xtest


# create model to use for training
def create_model(input_length):
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(input_length,), activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    # compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# evaluate a neural network model
def evaluate_model(Xtrain, ytrain, Xtest, ytest):
    scores = []
    n_repeats = 30
    n_words = Xtest.shape[1]
    model = create_model(n_words)
    desc = "Evaluation repeats"
    for i in tqdm(range(n_repeats), desc=desc):
        # fit network
        model.fit(Xtrain, ytrain, epochs=50, verbose=0)
        # evaluate
        loss, acc = model.evaluate(Xtest, ytest, verbose=0)
        scores.append(acc)
        # print("{} accuracy: {}".format(i+1, acc))
    return scores


# classify a review as negative (0) or positive (1)
def predict_sentiment(review, vocab, tokenizer, model):
    # clean
    tokens = clean_doc(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = " ".join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode="freq")
    # prediction
    yhat = model.predict(encoded, verbose=0)
    return round(yhat[0, 0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cv', "--cross_validate", action="store_true")
    parser.add_argument('-p', "--predict", action="store_true")
    args = parser.parse_args()
    # locate dataset
    script_dir = Path(__file__).resolve().parent
    dataset_dir = (
        script_dir / "../datasets/Movie Review Dataset/txt_sentoken"
    ).resolve()
    vocab_path = script_dir / "vocab.txt"
    # generate vocab if it doesn't exist
    if not vocab_path.exists():
        # define vocab
        print("Generating vocabulary")
        vocab = Counter()
        # add all docs to vocab
        process_docs(dataset_dir, vocab, is_train=True, generate_vocab=True)
        # print the size of the vocab
        print("Vocabulary size:", len(vocab))
        # print the top words in the vocab
        print("Most common words:", vocab.most_common(50))
        # keep tokens with a min occurrence
        min_occurane = 2
        tokens = [k for k, c in vocab.items() if c >= min_occurane]
        print(
            "Vocabulary size after removing words that appear fewer than {} times: {}".format(
                min_occurane, len(tokens)
            )
        )
        # save tokens to a vocabulary file
        save_list(tokens, vocab_path)

    # load the vocabulary
    print("Loading vocabulary")
    vocab = load_doc(vocab_path)
    vocab = vocab.split()
    vocab = set(vocab)

    # load all training reviews
    train_docs = process_docs(dataset_dir, vocab, is_train=True, generate_vocab=False)
    # load all test reviews
    test_docs = process_docs(dataset_dir, vocab, is_train=False, generate_vocab=False)
    # prepare labels
    ytrain = np.array([0 for _ in range(900)] + [1 for _ in range(900)])
    ytest = np.array([0 for _ in range(100)] + [1 for _ in range(100)])

    if args.cross_validate:
        modes = ["binary", "count", "tfidf", "freq"]
        results = pd.DataFrame()
        for mode in modes:
            # prepare data for mode
            Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
            # evaluate model on data for mode
            results[mode] = evaluate_model(Xtrain, ytrain, Xtest, ytest)
        # summarize results
        print(results.describe())
        # plot results
        results.boxplot()
        plt.show()

    if args.predict:
        # prepare data for training
        Xtrain, Xtest = prepare_data(train_docs, test_docs, "freq")
        # create the model
        model = create_model(Xtrain.shape[1])
        # fit network
        model.fit(Xtrain, ytrain, epochs=50, verbose=0)
        # create the tokenizer
        tokenizer = Tokenizer()
        # fit the tokenizer on the documents
        tokenizer.fit_on_texts(train_docs)
        while True:
            # get review to predict sentiment of
            review = input("Movie review: ")
            # evaluate model on data for mode
            print(predict_sentiment(review, vocab, tokenizer, model))
