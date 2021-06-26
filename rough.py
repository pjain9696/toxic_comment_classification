import yaml
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

test_mini = pd.read_csv('./data/test_mini.csv')
print(list(test_mini['comment_text']))
# model = tf.keras.models.load_model('./data/model_trainsize119678_modelparams3258214.h5')
# pickle.dump(model, open('./data/model_1.pkl', 'wb'))

# model = pickle.load(open('./data/model_1.pkl', 'rb'))
# print(model)

# print(os.path.isfile('./data/embed.csv'))

# with open('./config/config.yaml', 'r') as config_file:
#     config = yaml.safe_load(config_file)

# for class in config['preprocessing']['classes']:
#     print(class)
# print(config)
# print('\n\n\n')
# print(config.get('nn_params', None))

# classes = config['preprocessing']['classes']
# train = pd.read_csv('./data/train.csv')

# train_mini = train[:1000] #subsample train for speedy development, actual training will occue on full train set
# train_mini.to_csv('./data/train_mini.csv')
# print(train_mini.shape)

