#!/usr/bin/python
# -*- coding: utf-8 -*-
__version__ = "1.0"
__authors__ = "Jose Mar�a Alvarez"
__license__ = "MIT License <http://www.opensource.org/licenses/mit-license.php>"
__contact__ = "chema.ar@gmail.com"
__date__ = "2014-10-21"

import os
import sys
import time
import re
import urllib
import getopt
import collections
import itertools
import operator
import unittest
from igraph import *
from sklearn import datasets
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import random


def todot(g,fout):
   g.write_dot(fout)

def extract_graph_metrics(g):
   graph_features={}
   graph_features["vcount"] = g.vcount() 
   graph_features["ecount"] = g.ecount()
   graph_features["omega"] = g.omega()
   graph_features["alpha"] = g.alpha()
   return graph_features


def extract_graph_metrics_from_file(fin):
   g = Graph.Read_GraphML(fin, False)
   return extract_graph_metrics(g)
  
#From: http://igraph.wikidot.com/python-recipes

def laplacian_centrality(graph, vs=None):
   if vs is None:
       vs = xrange(graph.vcount())
   degrees = graph.degree(mode="all")
   result = []
   for v in vs:
       neis = graph.neighbors(v, mode="all")
       result.append(degrees[v]**2 + degrees[v] + 2 * sum(degrees[i] for i in neis))
   return result


def betweenness_centralization(G):
    vnum = G.vcount()
    if vnum < 3:
        raise ValueError("graph must have at least three vertices")
    denom = (vnum-1)*(vnum-2)
 
    temparr = [2*i/denom for i in G.betweenness()]
    max_temparr = max(temparr)
    return sum(max_temparr-i for i in temparr)/(vnum-1)

def create_svc_model(data, classification):
    clf = svm.SVC()
    clf.fit(data, classification) 
    return clf

def create_linear_svc_model(data, classification):
    clf = svm.LinearSVC()
    clf.fit(data, classification) 
    return clf

def classify_graph(clf, graph_instance):
   dec = clf.decision_function([graph_instance])
   return dec.shape[1]
   
if __name__ == "__main__":
   #Graphs
   graphs = {}
   # Load graph an external graph and extract characteristics
   graphs["graphs/primer.gml"] = extract_graph_metrics_from_file("graphs/primer.gml")
   #Create a random sample of graphs
   for i in range (0,5):
       graphs[i] = extract_graph_metrics(Graph.Erdos_Renyi(n=100, m=20))
   # Persist
   #foutput="graphs/primer.dot"
   #g.write_dot(foutput)
   #SVM: https://github.com/cjlin1/libsvm/tree/master/python
   sample_data = []
   sample_classification = []
   ngraphs = len(graphs.keys())
   for g in graphs.values():
       sample_data.append(g.values())
       sample_classification.append(random.randint(0,ngraphs))
   #print (sample_data)    
   #print (sample_classification)
   #Learn: a list of lists X where each row is the metric of a graph, and a list of classifications (0,1,2)
   #http://scikit-learn.org/stable/modules/svm.html#svm
   clf = create_svc_model(sample_data, sample_classification) 
   #Predict
   graph_to_predict = [2, 20, 100, 87]
   print classify_graph(clf,graph_to_predict)



    
    
    
    
