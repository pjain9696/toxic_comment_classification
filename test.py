import pandas as pd
import json
import yaml
import tensorflow as tf

#load classes from config
with open('./config/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
classes = config['preprocessing']['classes']

# load test sample
with open('./data/test_mini.txt') as f:
    test_mini = f.readlines()
test_mini = [x.strip('\n') for x in test_mini]
print(test_mini)
# test_mini = pd.read_csv('./data/test_mini.csv')
# test_mini = list(test_mini['comment_text'])

#convert to tokens
with open('./data/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
test_mini_tokenized = tokenizer.texts_to_sequences(test_mini)
test_mini_pad = tf.keras.preprocessing.sequence.pad_sequences(test_mini_tokenized, maxlen=200)

#load model
model_name = './data/model_trainsize119678_modelparams3258214.h5'
print('\nloading a saved model from disk...\n')
model = tf.keras.models.load_model(model_name)
pred_probs = model.predict(test_mini_pad)

for i in range(len(test_mini)):
    print('-'*50)
    print('comment = :', test_mini[i])
    print(dict(zip(classes, pred_probs[i])))
