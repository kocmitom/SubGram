#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "Tom Kocmi"

import logging
import Model
import time
import os
import word2vec


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
start = time.time()  # for counting the time

model = Model.getVectorModel() #create new model
#model = word2vec.Word2Vec.load("data/prepared.model")

#model.accuracy('data/questions.txt', restrict_vocab=30000)


print "Time: " + str(time.time() - start)
