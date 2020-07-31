
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse
from matplotlib.transforms import Bbox
import numpy as np
from matplotlib.dates import date2num
import pickle
import re
import datetime
from plotter import  linePlotter, whichAlg, doubleAxeslinePlotter



topologies = ['cycle', 'lollipop','geant', 'abilene','dtelekom', 'balanced_tree','grid_2d', 'hypercube','small_world','erdos_renyi']


topologies_varSize = {'cycle':'344', 'lollipop':'274', 'geant':'228', 'abilene':'85', 'dtelekom':'301', 'balanced_tree':'1434', 'grid_2d':'1665', 'hypercube':'1189', 'small_world':'1349', 'erdos_renyi':'1191'}
plt.show()
def whatVrSize( filename):
    "Find filename corresponds to which topology."
    for topology in topologies:
        if re.search(topology, filename):
            current_topology = topology
            return topologies_varSize[current_topology]

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description = 'Generate bar plots comparing different algorithms.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', type=str,
                   help='pickled file to be processed')
    parser.add_argument('--outfile', type=str, help="File to store the figure.")
    parser.add_argument('--normalize',action='store_true',help='Pass to normalize the plot.')
    parser.add_argument('--lgd',action='store_true',help='Pass to make a legened.')
    parser.add_argument('--plot_type', choices=['bar', 'line'], default='bar')
    args = parser.parse_args()
    
    
    DIC = {}
    
    filename = args.filename
    Alg  = whichAlg(filename)
    with open(filename) as current_file:
        alg_args, trace  = pickle.load(current_file)


    for iteration in trace:
        Key = str(trace[iteration]['time'])
        Val = trace[iteration]['OBJ'] * -1.0
        Val2 = trace[iteration]['CONSTRAINT']
        DIC[Key] =  (Val, Val2)
   
                

    
    print DIC
    doubleAxeslinePlotter(DIC, args.outfile, y_scale='linear', x_scale='linear' )
    


