import logging,datetime
import params
import numpy as np

def remove_common_words(list_words):
    keyword_list = ['a', 'the', 'of', 'an', 'at', 'for', 'from', 'on', 'to', 'or', 'and', '(', ')', 'it']
    for word in list_words:
        if word in keyword_list:
            list_words.remove(word)
    return list_words

def phrase_to_words(phrase):
    return remove_common_words(phrase.split())
    #return phrase.split()

def condition_vector(word_vecs):
    if len(word_vecs) > params.MAX_PHRASE_LEN:
        return word_vecs[:params.MAX_PHRASE_LEN]
    else:
        return word_vecs

def phraselist_to_2dvec(wordlist, word_actualvec_dict):
    if len(wordlist) < params.MAX_PHRASE_LEN:
        newvec = np.concatenate((np.array([wordvec(word, word_actualvec_dict) for word in wordlist]),
                                 np.zeros([params.MAX_PHRASE_LEN - len(wordlist), params.INPUT_SIZE])), axis=0)
    else:
        newvec = np.array([wordvec(word, word_actualvec_dict) for word in wordlist])
    return newvec

def wordvec(word, word_actualvec_dict):
    if word in word_actualvec_dict:
        return word_actualvec_dict[word]
    else:
        return np.zeros(params.INPUT_SIZE)

def createTrainingDataSet(word_actualvec_dict, meaninglist_word_list):
    meaningvecs_wordvec_list = []
    for val in meaninglist_word_list:
        wordlist, word = val[0], val[1]
        new2dvec = phraselist_to_2dvec(wordlist, word_actualvec_dict)
        meaningvecs_wordvec_list.append((new2dvec[:params.MAX_PHRASE_LEN, :], wordvec(word, word_actualvec_dict)))
    return meaningvecs_wordvec_list


def logInfo():
    timestamp = datetime.datetime.now().strftime('%d_%H_%M')
    filename_new = "results_" + params.MODEL_TYPE + "_" + timestamp + "_lr" + str(params.LEARNING_RATE) + "_hl" + str(
        params.HIDDEN_LAYERS) + ".log"
    print(filename_new)
    logging.basicConfig(filename=filename_new, level=logging.INFO)
    logging.info('Started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    logging.info("EMBEDDING_DIM        = {0}".format(params.EMBEDDING_DIM))
    logging.info("NUMRANKS             = {0}".format(params.NUMRANKS))
    logging.info("METRIC_TYPE          = {0}".format(params.METRIC_TYPE))
    logging.info("MODEL_TYPE           = {0}".format(params.MODEL_TYPE))
    logging.info("INPUT_SIZE           = {0}".format(params.INPUT_SIZE))
    logging.info("HIDDEN_LAYERS        = {0}".format(params.HIDDEN_LAYERS))
    logging.info("OUTPUT_SIZE          = {0}".format(params.OUTPUT_SIZE))
    logging.info("LEARNING_RATE        = {0}".format(params.LEARNING_RATE))
    logging.info("METRIC_THRESHOLD     = {0}".format(params.METRIC_THRESHOLD))
    logging.info("ITERATIONS_PER_EPOCH = {0}".format(params.ITERATIONS_PER_EPOCH))
    logging.info("BATCH_SIZE           = {0}".format(params.BATCH_SIZE))
    logging.info("EPOCHS               = {0}".format(params.EPOCHS))
    logging.info("MAX_PHRASE_LEN       = {0}".format(params.MAX_PHRASE_LEN))
    logging.info("REG_CONST            = {0}".format(params.REG_CONST))

#def get_word_meaning2dvec_map(word_actualvec_dict, parsed_dict):
#    word_meaning2dvec_map = {}
#    for word, wordlist in parsed_dict.items():
#        meaning2dvec = phraselist_to_2dvec(wordlist, word_actualvec_dict)
#        word_meaning2dvec_map[word] = meaning2dvec
#    return word_meaning2dvec_map