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

if __name__ == "__main__":
   # Load graph
   finput = "graphs/primer.gml"
   g = Graph.Read_GraphML(finput, False)
   # Extract characteristics
   graph_features={}
    # Basic measures
   summary(g)
   graph_features["vcount"] = g.vcount() 
   graph_features["ecount"] = g.ecount()
   graph_features["omega"] = g.omega()
   graph_features["alpha"] = g.alpha()
   # Persist
   foutput="graphs/primer.dot"
   g.write_dot(foutput)
   #SVM
  
    
    
    
    
    
