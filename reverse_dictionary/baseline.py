import numpy as np
import params
from model import Model

class Baseline_Model(Model):
    def __init__(self, word_actualvec_dict, meaningwordslist_word_list, metric_type):
        super().__init__(word_actualvec_dict, meaningwordslist_word_list, metric_type)
        #self.trainedvec_word_dict = word_actualvec_dict
        self.savetrainedvecdict()
        self.model = params.MODEL_BASELINE

    def calculateModelVec(self, input_phrase_2dvec):
        avg = np.average(input_phrase_2dvec, axis=0)
        return avg
