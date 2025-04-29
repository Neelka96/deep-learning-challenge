from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense

class Dense_Keras:
    def __init__(self, ml_data: dict, units: list[int], activations: list[str]):
        if len(units) != len(activations): raise ValueError('Lengths of layer units and activations must be equal. Exiting early.')
        self.output_activation = 'sigmoid'
        self.__parse_dict__(ml_data)
        self.__render_model__(units, activations)
        self.__save_layers__(units, activations)

    def __parse_dict__(self, ml_data: dict):
        self.input_dims = int(ml_data.get('input_dims'))
        self.output_dims = int(ml_data.get('output_dims'))
        self.X_train_scaled = ml_data.get('X_train_scaled')
        self.X_test_scaled = ml_data.get('X_test_scaled')
        self.y_train = ml_data.get('y_train')
        self.y_test = ml_data.get('y_test')
        return self

    def __render_model__(self, units, activations):
        self.model = Sequential()
        self.model.add(InputLayer(shape = (self.input_dims,)))
        [self.model.add(Dense(units = u, activation = a)) for u, a in zip(units, activations)]
        self.model.add(Dense(units = self.output_dims, activation = self.output_activation))
        return self
    
    def __save_layers__(self, units, activations):
        self.layers: list[tuple]
        self.layers = [(u, a) for u, a in zip(units, activations)]
        return self

    def summary(self):
        # Check the structure of the model
        return self.model.summary()
    
    def compile(
        self
        ,loss = 'binary_crossentropy'
        ,opt = 'adam'
        ,metrics = 'accuracy'
    ):
        if not isinstance(metrics, list): metrics = [metrics]
        
        self.model.compile(loss = loss, optimizer = opt, metrics = metrics)
        return self
    
    def train(self, epochs: int):
        self.model.fit(self.X_train_scaled, self.y_train, epochs = epochs)
        model_loss, model_accuracy = self.model.evaluate(self.X_test_scaled, self.y_test, verbose = 2)
        print(f'Accuracy: {model_accuracy}, Loss: {model_loss}')
        return self
    
    def params(self):
        ...

    def export_model(self, path: str):
        self.model.save(path)
        return None

# EOF

if __name__ == '__main__':
    print('Sorry, this module is for import only, not direct execution.')