import tensorflow as tf
import pandas as pd
import os

from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from utils import RocAucEvaluation
import module.model.model_specs as model_specs

class deep_model(object):
    def __init__(self, classes, vocab_size, embedding_matrix, params, logger):
        self.logger = logger
        self.vocab_size = vocab_size
        self.classes = classes
        self.params = params
        self.embedding_matrix = embedding_matrix
        self.model = self._build()
    
    def _build(self):
        model_name = self.params['model_name']
        embedding_layer = tf.keras.layers.Embedding(
            input_dim = self.vocab_size,
            output_dim = self.params['embedding_dim'],
            input_length = self.params['sentence_maxlen'],
            weights = [self.embedding_matrix],
            trainable = False
        )
        if model_name == 'gru':
            model = model_specs.GRU(embedding_layer, self.classes)
        elif model_name == 'lstm':
            model = model_specs.LSTM(embedding_layer, self.classes)
        elif model_name == 'CnnPlusGru_v1':
            model = model_specs.CnnPlusGru(embedding_layer, self.classes)
        else:
            raise Exception(f"model_name := {model_name} is not implemented yet")
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.logger.info(model.summary())
        return model
    
    def fit_and_validate(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        print('\n\nshape of train_x={}\nshape of train_y={}\n\n'.format(train_x.shape, train_y.shape))

        model_file = './data/model_{}_trainsize{}.h5'.format(self.model.name, len(train_x))
        if not os.path.exists(model_file):
            RocAuc = RocAucEvaluation(validation_data=(valid_x, valid_y), interval=1)
            history = self.model.fit(train_x, train_y, epochs=self.params['epochs'], verbose=True,
                validation_data=(valid_x, valid_y), callbacks=[RocAuc]
            )        
            #save model for reproducibility
            self.model.save(model_file)  # creates a HDF5 file
            df_pred_probs = self.predict_probs(self.model, test_x)
        else:
            print('\nloading a saved model from disk...\n')
            model = tf.keras.models.load_model(model_file)
            df_pred_probs = self.predict_probs(model, test_x)

        return df_pred_probs
    
    def predict_probs(self, model, test_x):
        pred_probs = model.predict(test_x)
        df_pred_probs = pd.DataFrame(pred_probs, columns=self.classes)
        return df_pred_probs