from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import pandas as pd
import numpy as np

class Dense_Keras:
    def __init__(self, hidden_layers: list[dict], **kwargs):
        input_dims = int(kwargs.get('input_dims'))
        output_dims = int(kwargs.get('output_dims'))
        self.X_train_scaled = kwargs.get('X_train_scaled')
        self.X_test_scaled = kwargs.get('X_test_scaled')
        self.y_train = kwargs.get('y_train')
        self.y_test = kwargs.get('y_test')
        self.model = Sequential()
        self.model.add(InputLayer(shape = (input_dims,)))
        [self.model.add(Dense(units = layer.get('units'), activation = layer.get('activation'))) for layer in hidden_layers]
        self.model.add(Dense(units = output_dims, activation = 'sigmoid'))


    def summary(self):
        # Check the structure of the model
        return self.model.summary()
    
    def compile(self, **kwargs):
        loss = kwargs.get('loss', 'binary_crossentropy')
        opt = kwargs.get('optimizer', 'adam')
        metrics = kwargs.get('metrics', 'accuracy')
        
        if not isinstance(metrics, list): metrics = [metrics]
        
        self.model.compile(loss = loss, optimizer = opt, metrics = metrics)
        return self
    
    def train(self, epochs: int):
        self.model.fit(self.X_train_scaled, self.y_train, epochs = epochs)
        model_loss, model_accuracy = self.model.evaluate(self.X_test_scaled, self.y_test, verbose = 2)
        print(f'Loss: {model_loss}, Accuracy: {model_accuracy}')
        return self

    def export_model(self, path: str):
        self.model.save(path)
        return None

# EOF

if __name__ == '__main__':
    print('Sorry, this module is for import only, not direct execution.')