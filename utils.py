import numpy as np
import logging
import yaml
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from tensorflow.keras.callbacks import Callback

def init_logger(config):
    if config['preprocessing']['dir_traindata'] == './data/train.csv':
        logger_file = 'model_history.log'
    else:
        logger_file = 'temp_model_history.log'
    
    logging.basicConfig(filename='logger_file', format='%(asctime)-15s %(message)s', level='INFO')
    logger = logging.getLogger('global_logger')

    logger.info('-' * 50)
    logger.info('')

    logger.info('config info:')
    for key, val in config.items():
        logger.info(key)
        logger.info(val)
    
    logger.info('')
    return logger

def load_config():
    with open('./config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def customAuc(yActual, yPred):
    fpr, tpr, thresholds = roc_curve(yActual, yPred)
    auc_score = auc(fpr, tpr)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    print('\nauc_score={}\noptimal_threshold={}\n'.format(auc_score, optimal_threshold))
    return auc_score, optimal_threshold

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.valid_x, self.valid_y = validation_data
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.valid_x, verbose=0)
            score = roc_auc_score(self.valid_y, y_pred)
            print('\n ROC-AUC for Validation - epoch: %d - score: %.6f\n' % (epoch+1, score))

