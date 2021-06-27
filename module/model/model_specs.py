import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPooling1D

def LSTM(embedding_layer, classes):
    model = tf.keras.Sequential(
        name = 'LSTM_v1',
        layers = [
            embedding_layer,
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(len(classes), activation='sigmoid'),
        ]
    )
    return model

def GRU(embedding_layer, classes):
    model = tf.keras.Sequential(
        name = 'GRU_v1',
        layers = [
            embedding_layer,
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(80, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
            tf.keras.layers.GlobalAveragePooling1D(),

            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(len(classes), activation='sigmoid'),
        ]
    )
    return model

def CnnPlusGru(embedding_layer, classes):
    model = tf.keras.Sequential(
        name = 'CnnPlusGru_v1',
        layers = [
        embedding_layer,
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        # tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(classes), activation='sigmoid')  #multi-label (k-hot encoding)
    ])
    return model