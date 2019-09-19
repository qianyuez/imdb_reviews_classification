# imdb_reviews_classification
Train different classifiers to solve imdb reviews classification task and compare their performance.


## Train
`cd imdb_reviews_classification`

`python main.py`

To get better result from keras model, set `validation_size` larger than 0 and set `plot` true. After training, you can get appropriate end step and prevent overfitting.


## Result
```
fullyConnectedNetwork:
test accuracy: 0.8924
test precision: 0.864722681497069
test recall: 0.9282026460148435
test f1 score: 0.8953388841335305

naiveBayesClassifier:
test accuracy: 0.85744
test precision: 0.8586744639376218
test recall: 0.8528557599225557
test f1 score: 0.8557552209810587

randomForestClassifier:
test accuracy: 0.74104
test precision: 0.7372215991027079
test recall: 0.7423362374959664
test f1 score: 0.7397700779805451

decisionTreeClassifier:
test accuracy: 0.73288
test precision: 0.69123745819398
test recall: 0.8336560180703453
test f1 score: 0.7557960944927961

kneighborsClassifier:
test accuracy: 0.59144
test precision: 0.704383664293743
test recall: 0.303323652791223
test f1 score: 0.4240442088643284
```
