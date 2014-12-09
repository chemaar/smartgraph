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
  
if __name__ == "__main__":
   # Load graph
   g = extract_graph_metrics_from_file("graphs/primer.gml")
   print (g)
   # Extract characteristics
   # Persist
   #foutput="graphs/primer.dot"
   #g.write_dot(foutput)
   #SVM: https://github.com/cjlin1/libsvm/tree/master/python
   data = g.values()
   target = g.keys()
   print (data)
   print(target)    

    
    
    
    
