import numpy as np
import math
import json


class NaiveBayes(object):
    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    def fit(self, X, y, k):
        print("========================")
        print("top-K: ", k)
        self.num_messages = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set(range(1, k+1, 1))

        self.num_messages['positive'] = sum(1 for label in y if label == 1)
        self.num_messages['negative'] = sum(1 for label in y if label == 0)

        n = len(X)
        self.log_class_priors['positive'] = math.log((self.num_messages['positive']) / n)
        self.log_class_priors['negative'] = math.log((self.num_messages['negative']) / n)


        self.word_counts['positive'] = {}
        self.word_counts['negative'] = {}

        for X, y in zip(X, y):
            c = 'positive' if y == 1 else 'negative'
            counts = self.get_word_counts(X)
            for word, count in counts.items():
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0

                self.word_counts[c][word] += count

        if k:
            self.word_counts["negative"] = {key: value for key, value in self.word_counts["negative"].items() if key <= k}
            self.word_counts["positive"] = {key: value for key, value in self.word_counts["positive"].items() if key <= k}

        # temp_dict = self.word_counts["negative"].copy()
        # temp_dict.update(self.word_counts["positive"])
        # self.vocab = set(list(temp_dict.keys()))

    def predict(self, X, y):
        result = []
        for x in X:
            counts = self.get_word_counts(x)
            positive = 0
            negative = 0
            for word, _ in counts.items():
                if word not in self.vocab:
                    continue

                # add Laplace smoothing
                log_positive = math.log(
                    (self.word_counts['positive'].get(word, 0.0) + 1) / (self.num_messages['positive'] + len(self.vocab)))
                log_negative = math.log(
                    (self.word_counts['negative'].get(word, 0.0) + 1) / (self.num_messages['negative'] + len(self.vocab)))

                positive += log_positive
                negative += log_negative

            positive += self.log_class_priors['positive']
            negative += self.log_class_priors['negative']

            if positive > negative:
                result.append(1)
            else:
                result.append(0)
        getAccuracy(y, result)
        getPrecision(y, result)
        getRecall(y, result)
        print("======================== \n")


# Accuracy = TP+TN/TP+FP+FN+TN
def getAccuracy(y, predictions):
    correct = 0
    for x in range(len(y)):
        if y[x] == predictions[x]:
            correct += 1
    print("Accuracy: ", "{0:.4f}".format((correct / float(len(y))) * 100.0))


# Precision = TP/TP+FP
def getPrecision(y, predictions):
    TP = 0
    FP = 0
    for x in range(len(y)):
        if y[x] == 1 and predictions[x] == 1:
            TP += 1
        if y[x] == 0 and predictions[x] == 1:
            FP += 1
    print("Precision: ", "{0:.4f}".format((TP / (TP+FP)) * 100.0))


# Recall = TP/TP+FN
def getRecall(y, predictions):
    TP = 0
    FN = 0
    for x in range(len(y)):
        if y[x] == 1 and predictions[x] == 1:
            TP += 1
        if y[x] == 1 and predictions[x] == 0:
            FN += 1
    print("Recall: ", "{0:.4f}".format((TP / (TP+FN)) * 100.0))


if __name__ == '__main__':
    X_test = np.load('./imdb/x_test.npy')
    X_train = np.load('./imdb/x_train.npy')
    y_test = np.load('./imdb/y_test.npy')
    y_train = np.load('./imdb/y_train.npy')
    # word_index_file = open('imdb_word_index.json')
    # word_str = word_index_file.read()
    # word_index = json.loads(word_str)
    # word_index_dict = dict(word_index)
    # word_index_sorted = { key: value for key, value in sorted(word_index_dict.items(), key=lambda item: (item[1], item[0]))}
    # print(word_index_sorted)
    # print(word_index_arr)

    NB = NaiveBayes()

    NB.fit(X_train, y_train, 100)
    NB.predict(X_test, y_test)

    NB.fit(X_train, y_train, 1000)
    NB.predict(X_test, y_test)

    NB.fit(X_train, y_train, 10000)
    NB.predict(X_test, y_test)




