import numpy as np
import yaml
import json
from utils import init_logger, load_config
from module.Preprocessor import Preprocessor
from module.trainer import  Trainer

if __name__ == "__main__":

    config = load_config()
    logger = init_logger(config)

    #preprocessing
    pp = Preprocessor(config, logger)
    train_x, train_y, valid_x, valid_y, test_x, test_y = pp.prep_data(
        load_pretrained_embeddings_from_disk=config['preprocessing']['load_pretrained_embeddings_from_disk']
    )
    print('\n\nfinished preparing data, shapes=\n')
    print('shape of train_x = {}, train_y = {}\n'.format(train_x.shape, train_y.shape))
    print('shape of valid_x = {}, valid_y = {}\n'.format(valid_x.shape, valid_y.shape))
    print('shape of test_x = {}, test_y = {}\n'.format(test_x.shape, test_y.shape))

    #training
    trainer = Trainer(config, logger, pp.classes, pp.vocab_size, pp.embedding_matrix)
    model, auc_dict, cls_report_dict = trainer.fit_and_validate(train_x, train_y, valid_x, valid_y, test_x, test_y)

    cls_report_dict = json.dumps(cls_report_dict, indent=2)

    print('Classification Report for TEST set per class: {}'.format(cls_report_dict))
    print('AUC Score for TEST set per class: {}'.format(auc_dict))
    print('Mean AUC Score for TEST set: {}'.format(np.mean(list(auc_dict.values()))))
    logger.info('AUC Score for TEST set per class: {}'.format(auc_dict))
    logger.info('\nClassification Report for TEST set per class: {}\n'.format(cls_report_dict))
