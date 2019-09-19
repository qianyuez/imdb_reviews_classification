from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class ClassifierGenerator():
    def __init__(self):
        self.classifiers = [
            MultinomialNB(),
            # SVC(random_state=1),
            DecisionTreeClassifier(random_state=1, criterion='gini'),
            RandomForestClassifier(random_state=1, criterion='gini'),
            KNeighborsClassifier(metric='minkowski')
        ]

        self.classifier_names = [
            'naiveBayesClassifier',
            # 'svc',
            'decisionTreeClassifier',
            'randomForestClassifier',
            'kneighborsClassifier',
        ]

        self.classifier_param_grid = [
            {'alpha': [0.1]},
            # {'C':[1], 'gamma':[0.01], 'kernel':('linear', 'rbf')},
            {'max_depth':[6,9,11,None]},
            {'n_estimators':[3,5,6]},
            {'n_neighbors':[4,6,8]},
        ]

    def train_classifiers(self, train_x, train_y):
        models = []
        for model, param in zip(self.classifiers, self.classifier_param_grid):
            print(model, param)
            classifier = GridSearchCV(model, param, cv=3)
            classifier.fit(train_x, train_y)
            models.append(classifier)
        return models, self.classifier_names
