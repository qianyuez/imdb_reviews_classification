from keras import models
from keras.layers import Dense, Dropout
from keras import optimizers, Sequential
import matplotlib.pyplot as plt


class BinaryClassificationModel():
    def __init__(self, input_size, optimizer='rmsprop'):
        super().__init__()
        self.input_size = input_size
        self.model = self._build_model(optimizer)

    def _build_model(self, optimizer):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(self.input_size,)))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
        return model

    def fit(self, train_x, train_y, validation_size=0.1, epochs=9, batch_size=32, plot=True):
        history = self.model.fit(train_x, train_y, validation_split=validation_size, epochs=epochs, batch_size=batch_size)
        if plot:
            self._plot_training_history(history.history)

    def predict(self, x):
        proba = self.predict_proba(x)
        prediction = [0 if p[0] < 0.5 else 1 for p in proba]
        return prediction

    def predict_proba(self, x):
        return self.model.predict(x)

    def _plot_training_history(self, history_dict):
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and Validation accuracy')
        plt.legend()
        plt.figure()

        plt.plot(epochs, loss, 'bo', label="Training loss")
        plt.plot(epochs, val_loss, 'b', label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.show()