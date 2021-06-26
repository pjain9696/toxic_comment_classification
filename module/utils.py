import numpy as np
import logging
import yaml
from sklearn.metrics import accuracy_score, roc_curve, auc

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

