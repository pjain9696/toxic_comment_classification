import json
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from utils import load_config
from module.contractions import expand_contractions

app = Flask(__name__)

#load classes from config
config = load_config()
classes = config['preprocessing']['classes']

#load model
model_name = config['app']['dir_final_model']
print('\nloading a saved model from disk...\n')
model = tf.keras.models.load_model(model_name)
print('model has been loaded\n')

#load tokenizer
tokenizer_file = config['app']['dir_tokenizer']
with open(tokenizer_file) as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    raw_input = [str.lower(x) for x in request.form.values()]
    expanded_input = expand_contractions(raw_input)     #expand contractions
    print(expanded_input)

    input_tokenized = tokenizer.texts_to_sequences(expanded_input)
    input_padded = tf.keras.preprocessing.sequence.pad_sequences(input_tokenized, maxlen=config['nn_params']['sentence_maxlen'])

    pred_probs = model.predict(input_padded)[0]
    pred_probs = [str(x) for x in pred_probs]
    output = dict(zip(classes, pred_probs))

    output = json.dumps(output, indent=2)
    print(output)

    return render_template(
        'index.html', 
        comment_text='You commented: {}'.format(raw_input[0]),
        prediction_text='Your comment has following toxictity score for each category (close to 1 means more toxic): {}'.format(output)
    )

if __name__ == "__name__":
    app.run()

app.run(port=5000)