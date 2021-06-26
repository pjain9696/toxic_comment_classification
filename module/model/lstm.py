import tensorflow as tf
import pandas as pd
import os

class LSTM(object):
    def __init__(self, classes, vocab_size, embedding_matrix, params, logger):
        self.logger = logger
        self.vocab_size = vocab_size
        self.classes = classes
        self.params = params
        self.embedding_matrix = embedding_matrix
        self.model = self._build()
    
    def _build(self):
        embedding_layer = tf.keras.layers.Embedding(
            input_dim = self.vocab_size,
            output_dim = self.params['embedding_dim'],
            input_length = self.params['sentence_maxlen'],
            weights = [self.embedding_matrix],
            trainable = False
        )
        model = tf.keras.Sequential([
            embedding_layer,
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(len(self.classes), activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.logger.info(model.summary())
        return model
    
    def fit_and_validate(self, train_x, train_y, valid_x, valid_y):
        print('\n\nshape of train_x={}\nshape of train_y={}\n\n'.format(train_x.shape, train_y.shape))

        model_name = './data/model_trainsize{}_modelparams{}.h5'.format(len(train_x), self.model.count_params())
        if not os.path.exists(model_name):
            history = self.model.fit(train_x, train_y, epochs=self.params['epochs'], verbose=True,
                validation_data=(valid_x, valid_y)
            )        
            #save model for reproducibility
            self.model.save(model_name)  # creates a HDF5 file
            df_pred_probs = self.predict_probs(self.model, valid_x)
        else:
            print('\nloading a saved model from disk...\n')
            model = tf.keras.models.load_model(model_name)
            df_pred_probs = self.predict_probs(model, valid_x)

        return df_pred_probs
    
    def predict_probs(self, model, test_x):
        pred_probs = model.predict(test_x)
        df_pred_probs = pd.DataFrame(pred_probs, columns=self.classes)
        return df_pred_probs