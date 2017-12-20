import numpy as np
from collections import defaultdict
import util, params, logging
import scipy.spatial.distance as ssd
from sklearn.preprocessing import normalize

class Model(object):
    def __init__(self, word_actualvec_dict, meaningwordslist_word_list, metric_type):
        self.word_actualvec_dict = word_actualvec_dict
        self.meaningwordslist_word_list = meaningwordslist_word_list
        self.metric_type = metric_type
        self.trainedvec_word_dict = []

    def train(self):
        # populate trainedvec_dict
        return NotImplementedError

    def calculateModelVec(self, input_phrase_2dvec):
        return NotImplementedError

    def getAccuracy(self, testdata, samplescount):
        correct_counts = 0
        for word, testphrase in testdata.items():
            predictions = self.predict_for_parsedPhrase(testphrase)
            logging.info('{0} ===== {1}'.format(word, ', '.join(predictions)))
            print(word , "====" ,  predictions)
            if word in predictions:
                correct_counts += 1
        acc = float(correct_counts) * 100.0 / samplescount
        print("accuracy: {0}%".format(acc))
        logging.info("********** accuracy: {0}% **********".format(acc))
        return acc

    def getMetricAndRank(self, input_vector):
        ranks = []
        #if len(self.trainedvec_word_dict) <10:
        #    self.savetrainedvecdict()
        for entry in self.trainedvec_word_dict:
            contextvector, word = entry[0], entry[1]
            if self.metric_type == params.METRIC_DOTPROD:
                metricval = np.dot(input_vector, contextvector)
            elif self.metric_type == params.METRIC_NORM:
                metricval = np.linalg.norm(contextvector - input_vector)
            elif self.metric_type == params.METRIC_COSINE:
                if (np.linalg.norm(input_vector) != 0 and np.linalg.norm(contextvector) != 0):
                    metricval = ssd.cosine(
                                    normalize(input_vector.reshape(-1,1), axis = 0),
                                    normalize(contextvector.reshape(-1,1), axis = 0))
                else: metricval = 0
            ranks.append((metricval, word))
        if self.metric_type == params.METRIC_DOTPROD:
            ranks.sort(reverse=True)
        elif self.metric_type == params.METRIC_NORM or self.metric_type == params.METRIC_COSINE:
            ranks.sort(reverse=False)
        else:
            RuntimeError('metric type undefined')
        return [x[1] for x in ranks[:params.NUMRANKS]]
        # , [x[0] for x in ranks[:params.NUMRANKS]]
        # return ranks[:params.NUMRANKS]

    def getCalculatedModelVec(self, phrase_2dvec):
        return self.calculateModelVec(util.condition_vector(phrase_2dvec))

    def getCalculatedModelVecForWordlist(self, wordlist):
        return self.calculateModelVec(util.phraselist_to_2dvec(wordlist, self.word_actualvec_dict))

    def predict_for_parsedPhrase(self,parsedPhrase):
        calculated_modelvect_for_phrase = self.getCalculatedModelVecForWordlist(parsedPhrase)
        return self.getMetricAndRank(calculated_modelvect_for_phrase)

    def predict_for_phrase(self, phrase):
        parsedPhrase = util.phrase_to_words(phrase)
        return self.predict_for_parsedPhrase(parsedPhrase)

    def savetrainedvecdict(self):
        for meaningwordlist_word_tuple in self.meaningwordslist_word_list:
            self.trainedvec_word_dict.append((self.getCalculatedModelVecForWordlist(meaningwordlist_word_tuple[0]),
                                             meaningwordlist_word_tuple[1]))




