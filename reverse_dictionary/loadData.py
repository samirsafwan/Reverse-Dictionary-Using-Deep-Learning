"""
Functions for reading and writing data to and from files.
"""

import os
import numpy as np
from collections import defaultdict
import util, params
import json

def parseDict( meaninglist_word_list, dictFile):
    if os.path.isfile(dictFile) == False:
        raise RuntimeError('Dictionary file not found')
    if meaninglist_word_list is None:
        meaninglist_word_list = []
    try:
        with open(dictFile) as dictData:
            for i in range(29):
                dictData.readline()
            for line in dictData.readlines():
                dictLine = line.split('|')
                word = dictLine[0].split()[4]
                phrase = dictLine[1].split(';')[0]
                definition = util.phrase_to_words(phrase)
                meaninglist_word_list.append((definition,word))
                if len(definition) > params.MAX_PHRASE_LEN:
                    params.MAX_PHRASE_LEN = len(definition)
    except:
        raise RuntimeError('Error detected in reading dictionary file')
    return meaninglist_word_list


def get_word_embeddings(path):

    """
    Load any embedding model written as text, in the format:
    word[space or tab][values separated by space or tab]

    :param path: path to embeddings file
    :return: a dict[word] = array.float(dim)
    """
    word_embedding = defaultdict(lambda: np.zeros(params.EMBEDDING_DIM))

    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            line = line.strip()
            if line == '':
                continue

            fields = line.split(' ')
            word = fields[0]
            vector = np.array([float(x) for x in fields[1:]], dtype=np.float32)
            word_embedding[word] = vector #/np.linalg.norm(vector) # normalize embeddings

    return word_embedding

def load_manual(path):
    if os.path.isfile(path) == False:
        raise RuntimeError('Dictionary file not found')
    testDict = defaultdict(list)
    count = 0
    try:
        with open(path) as file:
            for line in file.readlines():
                temp = line.split(':')
                templist = temp[1].split()
                newlist = []
                for word in templist:
                    word = word.replace(",","").replace("(","").replace(")","").replace(";","")
                    newlist.append(word)
                testDict[temp[0].replace(" ", "")] = newlist
                count += 1
    except:
        raise RuntimeError('Error detected in reading dictionary file')
    return testDict, count    

def load_test_data(path):
    '''
        Parses a WordNet data file and creates a dictionary of entries and definitions.
        Definitions are stored as lists of words.
        dictEntry = {word1: ['definition', 'of', 'word1', ...], word2: ...}
        '''
    if os.path.isfile(path) == False:
        raise RuntimeError('Dictionary file not found')
    testDict = defaultdict(list)
    count = 0
    try:
        with open(path) as testData:
            for i in range(29):
                testData.readline()
            for line in testData.readlines():
                count += 1
                dictLine = line.split('|')
                word = dictLine[0].split()[4]
                phrase = dictLine[1].split(';')[0]
                definition = util.phrase_to_words(phrase)
                if word not in testDict:
                    testDict[word] = definition
                else:
                    testDict[word] += definition
    except:
        raise RuntimeError('Error detected in reading dictionary file')

    return testDict, count


def load_trained_model(path):
    if os.path.isfile(path) == False:
        raise RuntimeError('Trained model file not found')
    try:
        with open(path) as modeldata:
            trained_model = modeldata
    except:
        raise RuntimeError('Error detected in reading trained model file')

    return trained_model

def getParsedDictAll():
    parsedDict = None
    parsedDict = parseDictOld(parsedDict, 'data/data.adj')
    parsedDict = parseDictOld(parsedDict, 'data/data.noun')
    parsedDict = parseDictOld(parsedDict, 'data/data.adv')
    parsedDict = parseDictOld(parsedDict, 'data/data.verb')
    return parsedDict

def parseDictOld(parsedDict, dictFile):
    '''
    Parses a WordNet data file and creates a dictionary of entries and definitions.
    Definitions are stored as lists of words.
    dictEntry = {word1: ['definition', 'of', 'word1', ...], word2: ...}
    '''
    if os.path.isfile(dictFile) == False:
        raise RuntimeError('Dictionary file not found')
    if parsedDict is None:
        parsedDict = defaultdict(list)
    try:
        with open(dictFile) as dictData:
            for i in range(29):
                dictData.readline()
            for line in dictData.readlines():
                dictLine = line.split('|')
                word = dictLine[0].split()[4]
                phrase = dictLine[1].split(';')[0]
                definition = util.phrase_to_words(phrase)
                if word not in parsedDict:
                    parsedDict[word] = definition
                else:
                    parsedDict[word] += definition
    except:
        raise RuntimeError('Error detected in reading dictionary file')

    return parsedDict