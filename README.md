# toxic_comment_classification

Identify toxicity in online comments.

## Dataset

Data for this project has been picked from Kaggle.

* Download data from [kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

* Unzip the files and save .csv in 'data/' folder

## Requirements

* Run following command to load required libraries from requirements.txt file:

```python
pip install -r requirements.txt
```

* Download pretrained GLoVe embeddings (glove.840B.300d) from [here](https://www.kaggle.com/takuok/glove840b300dtxt) or [here](https://nlp.stanford.edu/projects/glove/) and save to 'data/' folder.

* Ensure file names specified in config.yaml is consistent with your training and embedding file names

## Run script

* Choose preferable settings from [config.yaml](https://github.com/pjain9696/toxic_comment_classification/blob/master/config/config.yaml) before initiating traning:
  
  * load_pretrained_embeddings_from_disk has been defaulted to False, change to True if you want to avoid unpacking glove embeddings for each subsequent run
  
  * Update random_seed to maintain reproducibility of multiple experiments

  * run main.py
