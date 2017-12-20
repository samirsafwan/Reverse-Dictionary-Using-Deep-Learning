import os,sys, logging, datetime
import loadData, util, params
from lstm import LSTM_Model
from baseline import Baseline_Model
import attention, alignmentTable

def test(modelobj, testdatapath):
    testdata, samplescount = loadData.load_test_data(testdatapath)
    return modelobj.getAccuracy(testdata, samplescount)

def testmanual(modelobj):
    testdata, samplescount = loadData.load_manual(params.MANUAL_DATA_PATH)
    return modelobj.getAccuracy(testdata, samplescount)

def printUsage():
    print('Usage:')
    print('  python predict.py train')
    print('  python predict.py test <model_type>')
    print('  python predict.py test <model_type> <filepath>')
    print('  python predict.py test <model_type> <phrase or file with phrases>')
    sys.exit(0)

if len(sys.argv) < 2 :
    printUsage()
if len(sys.argv) == 2 or len(sys.argv) >= 3:
    params.MODEL_TYPE = params.MODEL_LSTM
if len(sys.argv) >= 3:
    params.MODEL_TYPE = sys.argv[2]

meaningwordslist_word_list = None
meaningwordslist_word_list = loadData.parseDict(meaningwordslist_word_list, 'data/data.adj')
meaningwordslist_word_list = loadData.parseDict(meaningwordslist_word_list, 'data/data.noun')
meaningwordslist_word_list = loadData.parseDict(meaningwordslist_word_list, 'data/data.adv')
meaningwordslist_word_list = loadData.parseDict(meaningwordslist_word_list, 'data/data.verb')
word_actualvec_dict = loadData.get_word_embeddings(params.GLOVE_FILEPATH)

print("MAX_PHRASE_LEN ", params.MAX_PHRASE_LEN)
################# update here only to change metric type ##########################
params.METRIC_TYPE = params.METRIC_NORM
util.logInfo()

if sys.argv[1] == "train":
    print("############################## training model ###############################")
    modelobj = LSTM_Model(word_actualvec_dict, meaningwordslist_word_list, params.METRIC_TYPE)
    modelobj.train()
    logging.info('Traing complete ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

elif sys.argv[1] == "test":
    if len(sys.argv)<3:
        printUsage()

    if sys.argv[2] == params.MODEL_LSTM:
        modelobj = LSTM_Model(word_actualvec_dict, meaningwordslist_word_list, params.METRIC_TYPE)
        modelobj.savedictTrained()
    elif sys.argv[2] == params.MODEL_BASELINE:
        modelobj = Baseline_Model(word_actualvec_dict, meaningwordslist_word_list, params.METRIC_TYPE)
    else:
        RuntimeError('model type undefined')

    logging.info('Test starts ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("############################## entering test ###############################")
    if len(sys.argv) == 3:
        test(modelobj, params.TEST_DATA_PATH)

    if len(sys.argv) == 4:
        if os.path.isfile(sys.argv[3]) == True:
            test(modelobj, sys.argv[3])
        elif sys.argv[3] == "man":
            testmanual(modelobj)
        elif sys.argv[3] == "play":
            parsedDict = loadData.getParsedDictAll()
            while input("press enter to play or enter 'exit' to quit!")  != "exit" :
                response = input("Please enter a phrase: ")
                print(response)
                inputphrase = util.phrase_to_words(response)
                output = modelobj.predict_for_phrase(response)
                print('Top 10 words that match the query:')
                for num, word in enumerate(output):
                    print('{}. {}'.format(num, word))
                chosenword = input("Type a chosen word to show alignment matrix: ")
                print("Creating alignment matrix against the following phrase: ")
                print(" ".join(parsedDict[chosenword]))
                a = util.phraselist_to_2dvec(parsedDict[chosenword], word_actualvec_dict)
                b = util.phraselist_to_2dvec(inputphrase, word_actualvec_dict)
                atil, btil = attention.attention(a,b)
                alignmentTable.alignmentTable(parsedDict[chosenword], inputphrase, atil, btil )
                print("Done!")
        else:
            phrase = sys.argv[3]
            output = modelobj.predict_for_phrase(phrase)
            print('Query - {}'.format(phrase))
            print('Top 10 words that match the query:')
            for num, word in enumerate(output):
                print('{}. {}'.format(num, word))
logging.info('Finished ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


