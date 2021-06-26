import json
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from module.utils import load_config

app = Flask(__name__)

#load classes from config
config = load_config()
classes = config['preprocessing']['classes']

#load model
model_name = './data/model_trainsize750_modelparams2346854.h5'
print('\nloading a saved model from disk...\n')
model = tf.keras.models.load_model(model_name)
print('\nmodel has been loaded\n')

#load tokenizer
with open('./data/tokenizer.json') as f:
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
    input = request.form.values()
    input = [x for x in request.form.values()]
    print(input)
    input_tokenized = tokenizer.texts_to_sequences(input)
    input_padded = tf.keras.preprocessing.sequence.pad_sequences(input_tokenized, maxlen=200)

    pred_probs = model.predict(input_padded)[0]
    pred_probs = [str(x) for x in pred_probs]
    output = dict(zip(classes, pred_probs))

    output = json.dumps(output, indent=2)
    print(output)

    return render_template(
        'index.html', 
        comment_text='You commented: {}'.format(input[0]),
        prediction_text='Your comment has the following toxicity probability score: {}'.format(output)
    )

if __name__ == "__name__":
    app.run()

app.run(port=5000)