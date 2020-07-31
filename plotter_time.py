
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
from plotter import  linePlotter, whichAlg



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
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                   help='pickled files to be processed')
    parser.add_argument('--outfile', type=str, help="File to store the figure.")
    parser.add_argument('--normalize',action='store_true',help='Pass to normalize the plot.')
    parser.add_argument('--lgd',action='store_true',help='Pass to make a legened.')
    parser.add_argument('--metric',choices=['OBJ','time'], default='time',help='Determine whether to plot gain or cost')
    parser.add_argument('--plot_type', choices=['bar', 'line'], default='bar')
    args = parser.parse_args()
    
    
    DICS = {}
    
    for filename in args.filenames:
        Alg  = whichAlg(filename)
        Key = whatVrSize(filename) #Key is the size of variables 
        with open(filename) as current_file:
            alg_args, trace  = pickle.load(current_file)
        trace_lastKey = max(trace.keys() )

        if Alg not in DICS:
             DICS[Alg] = {}

        try:
            DICS[Alg][Key] =  trace[trace_lastKey][args.metric]
        except KeyError:
            print filename
    #        continue
         
        if args.metric == 'OBJ':
            y_axis_label = 'Objective'
            DICS[Alg][Key] *= -1
            if args.normalize and Alg == 'LBSB':
                max_dict[Key] = DICS[Alg][Key]
        else:
            y_axis_label = 'Time (s)'

    #Normalize
    if args.normalize:
        for alg in DICS:
            for Key in DICS[alg]:
                print DICS[alg][Key], alg, Key
                DICS[alg][Key] /= max_dict[Key]
                print DICS[alg][Key] 

    
    print DICS
    linePlotter(DICS, args.outfile, yaxis_label=y_axis_label, xaxis_label='Number of variables',y_scale='log', x_scale='log' )
    


