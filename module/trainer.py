import pandas as pd
from sklearn.metrics import classification_report

from module.model.lstm import LSTM
from utils import customAuc

class Trainer:
    def __init__(self, config, logger, classes, vocab_size, embedding_matrix) -> None:
        self.config = config
        self.logger = logger
        self.classes = classes
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.model = LSTM(self.classes, self.vocab_size, self.embedding_matrix, self.config['nn_params'], self.logger)
    
    def fit_and_validate(self, train_x, train_y, valid_x, valid_y):
        df_valid_y_pred_probs = self.model.fit_and_validate(train_x, train_y, valid_x, valid_y)
        df_valid_y = pd.DataFrame(valid_y, columns=self.classes)
        auc_list, cls_report_list = self.metrics(df_valid_y_pred_probs, df_valid_y)
        return self.model, auc_list, cls_report_list
    
    def metrics(self, df_pred_probs, df_true_labels):
        auc_dict = {}
        cls_report_dict = {}
        for cl in self.classes:
            print('class =', cl)
            true_y_local, pred_y_local = df_true_labels[cl].tolist(), df_pred_probs[cl].tolist()
            auc_score, optimal_threshold = customAuc(true_y_local, pred_y_local)
            auc_dict[cl] = auc_score

            pred_labels = [1 if x> 0.5 else 0 for x in pred_y_local]
            cls_report = classification_report(true_y_local, pred_labels, zero_division=1, output_dict=True)
            cls_report_dict[cl] = cls_report

        
        return auc_dict, cls_report_dict