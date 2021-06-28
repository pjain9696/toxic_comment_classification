import pandas as pd
import numpy as np
import os
import io
import re
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from module.embedding import load_pretrained_embeddings
from module.contractions import contractions_list, expand_contractions

class Preprocessor:
    def __init__(self, config, logger) -> None:
        self.config = config['preprocessing']
        self.nn_params = config.get('nn_params', None)
        self.logger = logger
        self.classes = self.config['classes']
        self._load_raw_data()
        self.vocab_size = None #set in nn_vectorization method
        self.embedding_matrix = None #set in nn_vectorization method

    def _load_raw_data(self):
        self.df_train = pd.read_csv(self.config['dir_traindata'])
        self.df_test = pd.read_csv(self.config['dir_testdata'])
        test_labels_raw = pd.read_csv(self.config['dir_testlabels'])

        #few test samples are not evaluated, they have -1 label for all classes -> remove them from df_test
        test_labels_modified = test_labels_raw.copy()
        test_labels_modified['row_sum'] = test_labels_modified[self.config['classes']].sum(axis=1)
        test_labels_modified = test_labels_modified.loc[test_labels_modified['row_sum'] != -6]
        test_labels_modified.drop(['row_sum'], inplace=True, axis=1)
        print('\n\nshape of original test_labels = {}, shape of filtered test_labels_= {}\n\n'.format(
            test_labels_raw.shape, test_labels_modified.shape
        ))

        test_ids_to_keep = test_labels_modified['id']
        self.df_test = self.df_test.loc[self.df_test['id'].isin(test_ids_to_keep)]
        self.df_test_labels = test_labels_modified
        self.test_ids = self.df_test['id']
        return
    
    def prep_data(self, load_pretrained_embeddings_from_disk=False):
        data_x = self.df_train.comment_text.to_numpy()
        test_x = self.df_test.comment_text.to_numpy()
        
        print('\npreprocessing inputs:\n')
        data_x = expand_contractions(data_x)
        test_x = expand_contractions(test_x)

        train_x, train_y, valid_x, valid_y, test_x, test_y = self.nn_vectorization(data_x, test_x, 
            load_pretrained_embeddings_from_disk=load_pretrained_embeddings_from_disk
        )        
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def nn_vectorization(self, data_x, test_x, load_pretrained_embeddings_from_disk=False):
        params = self.nn_params
        
        data_y = self.df_train[self.classes]
        test_y = self.df_test_labels[self.classes].values

        train_x, valid_x, train_y, valid_y = train_test_split(
            data_x, data_y, test_size=0.2, random_state = self.config['random_seed']
        )

        #definitions
        num_tokens = params['num_tokens']
        maxlen = params['sentence_maxlen']
        # add a preprocessing step to train_x, test_x : remove 's / expand shortforms
        
        tokenizer = Tokenizer(num_words=num_tokens)
        tokenizer.fit_on_texts(train_x)

        #save tokenzier as json to be used during live (in production) predictions
        print('\nsaving tokenizer as json in disk...\n')
        tokenizer_json = tokenizer.to_json()
        with io.open(self.config['dir_tokenizer'], 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        
        self.vocab_size = min(num_tokens, len(tokenizer.word_index))

        train_x_tokenized = tokenizer.texts_to_sequences(train_x)
        train_x_pad = pad_sequences(train_x_tokenized, maxlen=maxlen)
        
        valid_x_tokenized = tokenizer.texts_to_sequences(valid_x)
        valid_x_pad = pad_sequences(valid_x_tokenized, maxlen=maxlen)

        test_x_tokenized = tokenizer.texts_to_sequences(test_x)
        test_x_pad = pad_sequences(test_x_tokenized, maxlen=maxlen)

        config = params.get('pretrained_embedding', None)
        '''
            load filtered embeddings from disk to save developer time (applicable from second iteration)
            ->  otherwise load full glove embeddings (memory expensive op) and store a filtered list in disk
                this shall save us time and computation effort on successive algo runs
        '''
        filtered_embed_path = './data/filtered_embed_vocabsize{}_dim{}.csv'.format(self.vocab_size, params['embedding_dim'])
        if load_pretrained_embeddings_from_disk and os.path.exists(filtered_embed_path):
            print('\nloading saved pretrained embeddings from disk, filename = {}...\n'.format(config['name']))       
            self.embedding_matrix = np.genfromtxt(filtered_embed_path, delimiter=',')
            print('\nloaded embedding matrix from path, shape =\n', self.embedding_matrix.shape)
        else:
            embeddings_index = load_pretrained_embeddings(config['file_path'])

            #initialize embedding_matrix with default values as mean of all embedding values
            self.embedding_matrix = np.zeros((self.vocab_size, params['embedding_dim']))
            print("\ncreating an embedding_matrix of shape =\n", self.embedding_matrix.shape)

            oov_words = []
            count = 0
            for word, i in tokenizer.word_index.items():
                if i >= self.vocab_size:
                    continue
                count += 1
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    self.embedding_matrix[i] = embedding_vector
                else:
                    oov_words.append(word)
            
            with open('./data/oov_words_vocabsize{}'.format(self.vocab_size), 'w') as file_handler:
                for item in oov_words:
                    file_handler.write("{}\n".format(item))

            print('V1--out of %d words in vocab, %d are missing from glove vocab\n' % (count, len(oov_words)))

            #save embedding matrix to file
            np.savetxt(filtered_embed_path, self.embedding_matrix, delimiter=",")
        
        return train_x_pad, train_y, valid_x_pad, valid_y, test_x_pad, test_y

