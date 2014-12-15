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
from os import listdir
from os.path import isfile, join


def todot(g,fout):
   g.write_dot(fout)

def extract_graph_metrics(g):
   graph_features={}
   #Basic measures: http://igraph.org/python/doc/igraph.GraphBase-class.html
   graph_features[g.vcount.__name__] = g.vcount() 
   graph_features[g.ecount.__name__] = g.ecount()
   graph_features[g.omega.__name__] = g.omega()
   graph_features[g.alpha.__name__] = g.alpha()
   graph_features[g.diameter.__name__] = g.diameter()
   graph_features[g.average_path_length.__name__] = g.average_path_length()
   graph_features[g.radius.__name__] = g.radius()
       
   #Structural properties
   graph_features["max_"+g.degree.__name__] = max(g.degree())
   graph_features["min_"+g.degree.__name__] = min(g.degree())
   graph_features["mean_"+g.degree.__name__] = np.mean(g.degree())
   graph_features["max_"+g.count_multiple.__name__] = max(g.count_multiple())
   graph_features["min_"+g.count_multiple.__name__] = min(g.count_multiple())
   graph_features["mean_"+g.count_multiple.__name__] = np.mean(g.count_multiple())
   graph_features[g.has_multiple.__name__] = g.has_multiple()
   graph_features[g.density.__name__] = g.density()
   graph_features["max_"+g.diversity.__name__] = max(g.diversity())
   graph_features["min_"+g.diversity.__name__] = min(g.diversity())
   graph_features["mean_"+g.diversity.__name__] = np.mean(g.diversity())
   graph_features["len_"+g.articulation_points.__name__] = len(g.articulation_points())
   graph_features[g.assortativity_degree.__name__] = g.assortativity_degree()
       

   #Centrality
   graph_features["max_laplacian_centrality"] = max(laplacian_centrality(g))
   graph_features["min_laplacian_centrality"] = min(laplacian_centrality(g))
   graph_features["mean_laplacian_centrality"] = np.mean(laplacian_centrality(g))
   graph_features["betweenness_centralization"] = betweenness_centralization(g)
   betweeness_per_node=g.betweenness()
   graph_features["max_"+g.edge_betweenness.__name__] = max(g.edge_betweenness())
   graph_features["min_"+g.edge_betweenness.__name__] = min(g.edge_betweenness())
   graph_features["mean_"+g.edge_betweenness.__name__] = np.mean(g.edge_betweenness())
   graph_features["max_"+g.closeness.__name__] = max(g.closeness())
   graph_features["min_"+g.closeness.__name__] = min(g.closeness())
   graph_features["mean_"+g.closeness.__name__] = np.mean(g.closeness())
   graph_features["max_in_"+g.closeness.__name__] = max(g.closeness(mode="in"))
   graph_features["min_in_"+g.closeness.__name__] = min(g.closeness(mode="in"))
   graph_features["mean_in_"+g.closeness.__name__] = np.mean(g.closeness(mode="in"))
   graph_features["max_out_"+g.closeness.__name__] = max(g.closeness(mode="out"))
   graph_features["min_out_"+g.closeness.__name__] = min(g.closeness(mode="out"))
   graph_features["mean_out_"+g.closeness.__name__] = np.mean(g.closeness(mode="out"))
   graph_features[g.canonical_permutation.__name__] = len(g.canonical_permutation())
   graph_features[g.clique_number.__name__] = g.clique_number()
   #graph_features[g.largest_cliques.__name__] = g.largest_cliques()

   #Clustering
   graph_features[g.count_automorphisms_vf2.__name__] = g.count_automorphisms_vf2()
   graph_features["len_"+g.cut_vertices.__name__] = len(g.cut_vertices())
   graph_features["len_"+g.knn.__name__] = len(g.knn())
   graph_features["len_"+g.biconnected_components.__name__] = len(g.biconnected_components())

   #clusters=g.clusters() #weak or strong
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
def load_graphs_from_dir(dir):
   train_graphs = {}
   onlyfiles = [ f for f in listdir(dir) if isfile(join(dir,f)) ]
   for f in onlyfiles:
      print ("Loading: "+f+" from "+join(dir,f))
      train_graphs[f] = extract_graph_metrics_from_file(join(dir,f))
   return train_graphs
   
if __name__ == "__main__":
   #Graphs
   train_dir="./graphs/train/"
   test_dir="./graphs/test/"
   # Load graph an external graph and extract characteristics
   train_graphs=load_graphs_from_dir(train_dir)
   # Persist
   #SVM: https://github.com/cjlin1/libsvm/tree/master/python
   max_classes = 10
   sample_data = []
   sample_classification = []
   for g in train_graphs.values():
       sample_data.append(g.values())
       sample_classification.append(random.randint(0,max_classes))
   #Learn: a list of lists X where each row is the metric of a graph, and a list of classifications (0,1,2)
   #http://scikit-learn.org/stable/modules/svm.html#svm
   clf = create_svc_model(sample_data, sample_classification) 
   #Predict
   test_graphs=load_graphs_from_dir(test_dir)
   for g in test_graphs.values():
        print classify_graph(clf,g.values())
        print clf.predict(g.values())



    
    
    
    
