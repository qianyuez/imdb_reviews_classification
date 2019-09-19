from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

class ModelEvaluator():
    def __init__(self, models, model_names, test_x, test_y):
        self.models = models
        self.model_names = model_names
        self.test_x = test_x
        self.test_y = test_y

    def compute_model_scores(self):
        for model, name in zip(self.models, self.model_names):
            y_pred = model.predict(self.test_x)
            accuracy = accuracy_score(self.test_y, y_pred)
            precision = precision_score(self.test_y, y_pred)
            recall = recall_score(self.test_y, y_pred)
            f1 = f1_score(self.test_y, y_pred)
            print(name + ':')
            print('test accuracy: {}'.format(accuracy))
            print('test precision: {}'.format(precision))
            print('test recall: {}'.format(recall))
            print('test f1 score: {}'.format(f1))
            print('')

    def compare_models(self):
        plt.figure()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve comparison')
        for model, name in zip(self.models, self.model_names):
            y_prob = model.predict_proba(self.test_x)
            if not isinstance(y_prob, np.ndarray):
                y_prob = np.array(y_prob)
            y_score = y_prob[:, -1]
            fpr, tpr, _ = roc_curve(self.test_y, y_score)
            auc_value = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='{} auc: {:.2f}'.format(name, auc_value))
        plt.legend()
        plt.show()


