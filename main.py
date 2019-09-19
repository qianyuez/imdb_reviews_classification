from load_data import load_data
from classifier_generator import ClassifierGenerator
from binary_classification_model import BinaryClassificationModel
from model_evaluate import ModelEvaluator

if __name__ == '__main__':
    onehot_train, onehot_test, y_train, y_test = load_data('./data/IMDB Dataset.csv')
    classifiers, classifier_names = ClassifierGenerator().train_classifiers(onehot_train, y_train)
    model = BinaryClassificationModel(input_size=onehot_train.shape[-1])
    print('start training fully connected network')
    model.fit(onehot_train, y_train, epochs=5, validation_size=0, plot=False)

    classifier_evaluator = ModelEvaluator(classifiers + [model], classifier_names + ['fullyConnectedNetwork'], onehot_test, y_test)
    classifier_evaluator.compute_model_scores()
    classifier_evaluator.compare_models()
