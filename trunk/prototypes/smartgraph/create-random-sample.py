#!/usr/bin/python
# -*- coding: utf-8 -*-
__version__ = "1.0"
__authors__ = "Jose Mar�a Alvarez"
__license__ = "MIT License <http://www.opensource.org/licenses/mit-license.php>"
__contact__ = "chema.ar@gmail.com"
__date__ = "2014-10-21"

import os
import sys
from igraph import *


def todot(g,fout):
   g.write_dot(fout)
   
def togml(g,fout):
   g.write_gml(fout)
   
def tographml(g,fout):
   g.write_graphml(fout)

if __name__ == "__main__":
   #Create  a set of graphs
   for i in range (0,5):
      g = Graph.Erdos_Renyi(n=100, m=20)
      todot(g, "graphs/train/graphs"+str(i)+".dot")
      #togml(g, "graphs/train/graphs"+str(i)+".gml")
      tographml(g, "graphs/train/graphs"+str(i)+".gml")
      

   
  



    
    
    
    
