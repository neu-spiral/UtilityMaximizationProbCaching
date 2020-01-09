
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



colors =['b', 'g', 'r', 'c' ,'m' ,'y' ,'k' ,'w']
hatches = ['////', '/', '\\', '\\\\', '-', '--', '+', '']


Algorithms = ['Barrier', 'Greedy1','Greedy2']
graph2lbl =  {'erdos_renyi':'ER','special_case':'path','erdos_renyi2':'ER-20Q','hypercube2':'hypercube-20Q'}
    
topologies = ['balanced_tree','cycle', 'lollipop','erdos_renyi', 'grid_2d','star','hypercube','small_world','barabasi_albert', 'dtelekom','abilene','geant']
def whichAlg( filename):
    "Find filename corresponds to which algrotihm."
    if re.search('Greedy1',  filename ):
        return 'Greedy1'
    elif re.search('Greedy2',  filename ):
        return 'Greedy2'
    else:
        return 'Barrier' 
def whichTopology( filename):
    "Find filename corresponds to which topology."
    for topology in topologies:
        if re.search(topology, filename):
            return topology
    

def barPlotter(DICS, outfile, y_axis_label = 'Objective'):
    def form_vals(DICS_alg):
        out = []
        labels = []
        for topology in topologies:
            try:
                out.append( DICS_alg[topology]  )
                labels.append(topology)
            except KeyError:
                print topology
                continue 
        return out, labels
        
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 6)
    width = 1


    N = len(DICS[Algorithms[0]].keys()) 
    numb_bars = len(Algorithms) 
    ind = np.arange(0,numb_bars*N ,numb_bars)
    print ind
    RECTS = []
    i = 0
    for alg in Algorithms:
        values, labels = form_vals(DICS[alg])
        print len(ind), len(values), N
        ax.bar(ind+i*width, values, align='center',width=width, color = colors[i], hatch = hatches[i],label=alg,log=True)
        RECTS+= ax.bar(ind+i*width, values, align='center',width=width, color = colors[i], hatch = hatches[i],label=alg,log=True)
        i+=1
    if args.lgd:
        LGD = ax.legend(Algorithms, ncol=len(Algorithms),borderaxespad=0.,loc=3, bbox_to_anchor=(0., 1.02, 1., .102),fontsize=15,mode='expand')    
    else:
        LGD = None
    
    
    ax.set_xticks(ind) 
    ax.set_xticklabels(tuple(labels),fontsize = 18)
    ax.set_ylabel(y_axis_label,fontsize=18)
  #  ax.set_yticklabels([0, 0.5, 1])
    plt.yticks(fontsize = 18)
    plt.xlim([ind[0]-width,ind[-1]+len(Algorithms)*width])
        
    fig_size = plt.rcParams["figure.figsize"]
    if args.lgd:
        fig.savefig(outfile+".pdf",format='pdf', bbox_extra_artists=(LGD,), bbox_inches=Bbox(np.array([[0,0],[18,6]])) )
    else:
        fig.savefig(outfile+".pdf",format='pdf', bbox_inches=Bbox(np.array([[0,0],[18,6]])) )
    plt.show()    
def read_file(fname,normalize=0):
    f = open(fname,'r')
    l= eval(f.readline())
    f.close
    (Time, OBJ) = l[-1] 
    return {"TIME":Time,"OBJ":OBJ}
def CG_readfile(CG_file,rounded_file):
    dic_CG = read_file(CG_file)
    dic_round = read_file(rounded_file)
    return {"TIME":dic_CG["TIME"]+dic_round["TIME"],"OBJ":dic_round["OBJ"]}
def gen_dictionaries(random_f,greedy_f,CG_1,round_1,CG_2,round_2,CG_3,round_3):
    dic_time = {"Random":0.,"Greedy":0.,"CG (10 samples)":0.,"CG (100 samples)":0.}
    dic_obj = {"Random":0.,"Greedy":0.,"CG (10 samples)":0.,"CG (100 samples)":0.}
    dic_time["Random"] = read_file(random_f)["TIME"]
    dic_obj["Random"] = read_file(random_f)["OBJ"] 
    
    dic_time["Greedy"] = read_file(greedy_f)["TIME"]
    dic_obj["Greedy"] = read_file(greedy_f)["OBJ"] 
    
    dic_time["CG (10 samples)"] = CG_readfile(CG_1,round_1)["TIME"]
    dic_obj["CG (10 samples)"] = CG_readfile(CG_1,round_1)["OBJ"] 
    dic_time["CG (100 samples)"] = CG_readfile(CG_2,round_2)["TIME"]
    dic_obj["CG (100 samples)"] = CG_readfile(CG_2,round_2)["OBJ"]
    dic_time["CG (200 samples)"] = CG_readfile(CG_3,round_3)["TIME"]
    dic_obj["CG (200 samples)"] = CG_readfile(CG_3,round_3)["OBJ"]
    return dic_time,dic_obj

plt.show()
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description = 'Generate bar plots comparing different algorithms.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                   help='pickled files to be processed')
    parser.add_argument('--outfile', type=str, help="File to store the figure.")
    parser.add_argument('--lgd',action='store_true',help='Pass to make a legened.')
    parser.add_argument('--metric',choices=['OBJ','time'], default='OBJ',help='Determine whether to plot gain or cost')
    parser.add_argument('--congestion', type=float, default=0.95, help="Congestion of the network")
    args = parser.parse_args()
    
    
    DICS = {}
    

    for filename in args.filenames:
        Alg  = whichAlg(filename)
        topology = whichTopology(filename)
        with open(filename) as current_file:
            alg_args, trace  = pickle.load(current_file)
        
        trace_lastKey = max(trace.keys() )
        if Alg not in DICS:
             DICS[Alg] = {}

        DICS[Alg][topology] =  trace[trace_lastKey ][args.metric]
         
        if args.metric == 'OBJ':
            y_axis_label = 'Objective'
            DICS[Alg][topology] *= -1
        else:
            y_axis_label = 'Time(s)'
    barPlotter(DICS, args.outfile, y_axis_label)               
    


