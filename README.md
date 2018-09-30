# Naive Bayes

## Requirements
- Numpy
- Math

## Initialize
First Load all the files to respective variables
```python
X_test = np.load('./imdb/x_test.npy')
X_train = np.load('./imdb/x_train.npy')
y_test = np.load('./imdb/y_test.npy')
y_train = np.load('./imdb/y_train.npy')
```

Create our NaiveBayes Class and do fitting and predicting
```python
NB = NaiveBayes()

NB.fit(X_train, y_train, 100)
NB.predict(X_test, y_test)

NB.fit(X_train, y_train, 1000)
NB.predict(X_test, y_test)

NB.fit(X_train, y_train, 10000)
NB.predict(X_test, y_test)
```

## Fitting
By creating the class we can use some class variables that may be save for later use

`num_messages` : Object to save the total numbers of Positive and Negative comments

`log_class_priors` : Get the log value of positive or negative comments divided by total comments

`vocab` : The top-K (100, 1000, 10000) frequently used words will be stored here

Remove words that are not in vocab where vocab = `range of k`
 ```python
self.word_counts["negative"] = {key: value for key, value in self.word_counts["negative"].items() if key <= k}
self.word_counts["positive"] = {key: value for key, value in self.word_counts["positive"].items() if key <= k}
```

## Predicting
Predicting is pretty simple because this method will use `Laplace smoothing`.

To start predicting we loop thru `each word`.

#### Laplace smoothing
- Adding up sum of positive and negative words log value and find the greater value
    - ex. If `positive is greater than negative` the comment is determined as `positive` and vice-versa.
- wordcount + `1`
    - when wordcount is 0, log will return error
    - therefore adding 1 to prevent log error
- reuse the `num_messages[positive and negative]` + total `vocab or k`
    - I didn't +1 in the denominator because, I skipped the words that are not in the vocab or k
```python
# add Laplace smoothing
log_positive = math.log(
    (self.word_counts['positive'].get(word, 0.0) + 1) / (self.num_messages['positive'] + len(self.vocab)))
log_negative = math.log(
    (self.word_counts['negative'].get(word, 0.0) + 1) / (self.num_messages['negative'] + len(self.vocab)))
```

#### Accuracy
TP+TN/TP+FP+FN+TN
- A ratio of correctly predicted observation to the total observations
-  Determine as good when `above 0.8`
```python
def getAccuracy(y, predictions):
    correct = 0
    for x in range(len(y)):
        if y[x] == predictions[x]:
            correct += 1
    print("Accuracy: ", "{0:.4f}".format((correct / float(len(y))) * 100.0))
```

#### Precision
TP/TP+FP 
- A ratio of correctly predicted positive observations to the total predicted positive observations
-  Determine as good when `above 0.7`
```python
def getPrecision(y, predictions):
    TP = 0
    FP = 0
    for x in range(len(y)):
        if y[x] == 1 and predictions[x] == 1:
            TP += 1
        if y[x] == 0 and predictions[x] == 1:
            FP += 1
    print("Precision: ", "{0:.4f}".format((TP / (TP+FP)) * 100.0))
```

#### Recall 
TP/TP+FN
-  A ratio of correctly predicted positive observations to the all observations in actual class
-  Determine as good when `above 0.5`
```python
def getRecall(y, predictions):
    TP = 0
    FN = 0
    for x in range(len(y)):
        if y[x] == 1 and predictions[x] == 1:
            TP += 1
        if y[x] == 1 and predictions[x] == 0:
            FN += 1
    print("Recall: ", "{0:.4f}".format((TP / (TP+FN)) * 100.0))
```
