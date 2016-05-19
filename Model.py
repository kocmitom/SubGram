#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Tom Kocmi"

import os.path
from time import time

import re
import CONS
import utils
import word2vec
import codecs


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    #return unicode(text.replace('\xc2\x85', '<newline>'), encoding, errors=errors)
    return unicode(text, encoding, errors=errors)


utils.to_unicode = any2unicode


def parseCorpus(line):
    # return re.sub('[0-9]+', '0', re.sub('[^a-z0-9 ]+', ' ', line.lower()))  # for english
    # return re.sub('[^a-z ]+', '', line.lower()) #for english
    return line


def getVectorModel():
    class MySentences(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            if CONS.ISINDIRECTORY:
                for fname in os.listdir(self.dirname):
                    if not fname.endswith('.txt'):
                        continue
                    for line in open(os.path.join(self.dirname, fname)):
                        yield parseCorpus(line).split()
            else:
                for line in open(self.dirname):
                    yield parseCorpus(line).split()

    sentences = MySentences(CONS.CORPUSNAME) # a memory-friendly iterator
    # # memory exhaustive way
    # with codecs.open(CONS.CORPUSNAME, "r") as myfile:
    #     documents = myfile.read().decode("utf-8").lower().split()
    #     sentences = [word.split() for word in documents]
    # model = word2vec.Word2Vec(sentences, min_count=min_count_words, size=sizeNN, workers=cpu)

    model = word2vec.Word2Vec()  # an empty model, no training
    model.min_count = CONS.MINWORDCOUNT
    model.vector_size = CONS.NNSIZE
    model.layer1_size = CONS.NNSIZE
    model.workers = CONS.CPU
    model.subgram = True
    model.min_count_sub = CONS.MINSUBSTRINGS
    model.build_vocab(sentences)

    model.train(sentences)
    model.save('data/' + str(time()) + "data.model")
    return model
