#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.svm import SVC
from modAL.models import ActiveLearner
from modAL.uncertainty import margin_sampling
from modAL.uncertainty import entropy_sampling

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


# In[ ]:


def train_dev_split(data, labels, labeledPool_size):
    n_labeled_examples = data.shape[0]
    training_indices = np.random.randint(low=0, high=n_labeled_examples - 1, size=labeledPool_size)
    X_train = data[training_indices]
    y_train = labels[training_indices]
    X_pool = np.delete(data, training_indices, axis=0)
    y_pool = np.delete(labels, training_indices, axis=0)
    return X_train, y_train, X_pool, y_pool

def train_dev_test_split(data, labels, labeledPool_size, test_size):
    n_labeled_examples = data.shape[0]
    training_indices = np.random.randint(low=0, high=n_labeled_examples - 1, size=labeledPool_size)
    X_train = data[training_indices]
    y_train = labels[training_indices]
    X_pool = np.delete(data, training_indices, axis=0)
    y_pool = np.delete(labels, training_indices, axis=0)
    test_indices = np.random.randint(low=0, high=n_labeled_examples - labeledPool_size- 1, size=test_size)
    X_test = data[test_indices]
    y_test = labels[test_indices]
    X_pool = np.delete(data, test_indices, axis=0)
    y_pool = np.delete(labels, test_indices, axis=0)
    return X_train, y_train,X_test, y_test, X_pool, y_pool

def displayAddressPair(index, addressPairs):
    lines_indices = ['Address 1', 'Address 2']
    columns = ['INBUILDING','EXTBUILDING','POILOGISTIC','ZONE','HOUSENUM','ROADNAME', 'CITY']
    valuesAdd1, valuesAdd2 = [], []
    for i in range (19, 26):
        valuesAdd1.append(addressPairs[index][i])
    for i in range (26, 33):
        valuesAdd2.append(addressPairs[index][i])
    values = [valuesAdd1, valuesAdd2]
    pair = pd.DataFrame(values, index = lines_indices, columns = columns )
    display(pair)
    

def manualAL(classifier, sampleRequest, nbIterations, X_train, y_train, X_pool, y_pool, X_poolWithAdd):
    learner = ActiveLearner(estimator=classifier, 
                            query_strategy = sampleRequest,
                            X_training=X_train, 
                            y_training=y_train)
    model_accuracy = learner.score(X_pool, y_pool)
    performance_history = [model_accuracy]
    for index in range(nbIterations):
        query_index, query_instance = learner.query(X_pool)
        X = X_pool[query_index]
        afficherDonnee(int(query_index), X_poolWithAdd)
        y = [float(input('Labellisation : 2: Match , 1: PartialMatch, 0: NoMatch \n'))]
        print(y_pool[query_index])
        learner.teach(X=X, y=y)
        X_pool,  y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index, axis=0)
        X_poolWithAdd = np.delete(X_poolWithAdd, query_index, axis=0)
        model_accuracy = learner.score(X_pool, y_pool)
        performance_history.append(model_accuracy)
    return performance_history

def manualAL_test(classifier, sampleRequest, nbIterations, X_train, y_train,X_test, y_test, X_pool, y_pool, X_poolWithAdd):
    learner = ActiveLearner(estimator=classifier, 
                            query_strategy = sampleRequest,
                            X_training=X_train, 
                            y_training=y_train)
    model_accuracy = learner.score(X_test, y_test)
    performance_history = [model_accuracy]
    for index in range(nbIterations):
        query_index, query_instance = learner.query(X_pool)
        X = X_pool[query_index]
        afficherDonnee(int(query_index), X_poolWithAdd)
        y = [float(input('Labellisation : 2: Match , 1: PartialMatch, 0: NoMatch \n'))]
        learner.teach(X=X, y=y)
        X_pool,  y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index, axis=0)
        X_poolWithAdd = np.delete(X_poolWithAdd, query_index, axis=0)
        model_accuracy = learner.score(X_test, y_test)
        performance_history.append(model_accuracy)
    return performance_history

def autoAL(classifier, sampleRequest, nbIterations, X_train, y_train, X_pool, y_pool):
    learner = ActiveLearner(estimator=classifier, 
                        query_strategy = sampleRequest,
                        X_training=X_train, 
                        y_training=y_train)
    model_accuracy = learner.score(X_pool, y_pool)
    performance_history = [model_accuracy]
    for index in range(nbIterations):
        query_index, query_instance = learner.query(X_pool)
        X, y = X_pool[query_index], y_pool[query_index].reshape(1, )
        learner.teach(X=X, y=y)
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
        model_accuracy = learner.score(X_pool, y_pool)
        performance_history.append(model_accuracy)
    return performance_history


def autoAL_test(classifier, sampleRequest, nbIterations, X_train, y_train,X_test, y_test, X_pool, y_pool):
    learner = ActiveLearner(estimator=classifier, 
                        query_strategy = sampleRequest,
                        X_training=X_train, 
                        y_training=y_train)
    model_accuracy = learner.score(X_test, y_test)
    performance_history = [model_accuracy]
    for index in range(nbIterations):
        query_index, query_instance = learner.query(X_pool)
        X, y = X_pool[query_index], y_pool[query_index].reshape(1, )
        learner.teach(X=X, y=y)
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
        model_accuracy = learner.score(X_test, y_test)
        performance_history.append(model_accuracy)
    return performance_history

