import numpy as np
import logging
from sklearn import metrics
from sklearn.metrics import f1_score

from pytorch_utils import forward
from utilities import get_filename
import config


def calculate_accuracy(y_true, y_score):
    N = y_true.shape[0]
    accuracy = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_score, axis=-1)) / N
    return accuracy


class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader):

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        cm = metrics.confusion_matrix(np.argmax(target, axis=-1), np.argmax(clipwise_output, axis=-1), labels=None)
        accuracy = calculate_accuracy(target, clipwise_output)

        f1 = f1_score(np.argmax(target, axis=-1), np.argmax(clipwise_output, axis=-1), average='macro')

        f1_score_per_class = f1_score(np.argmax(target, axis=-1), np.argmax(clipwise_output, axis=-1), average=None)
        print(f1_score_per_class)

        statistics = {'accuracy': accuracy, 'f1': f1}

        return statistics