
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
lin_styles = ['b^-','g*--','rD--','cX-','m*--','y-']


Algorithms = ['LBSB', 'CR','Greedy1','Greedy2']
graph2lbl =  {'erdos_renyi':'ER','special_case':'path','erdos_renyi2':'ER-20Q','hypercube2':'hypercube-20Q'}
    
topologies = ['cycle', 'lollipop','geant', 'abilene','dtelekom', 'balanced_tree','grid_2d', 'hypercube','small_world','erdos_renyi']

def whichAlg( filename):
    "Find filename corresponds to which algrotihm."
    if re.search('Greedy1',  filename ):
        return 'Greedy1'
    elif re.search('Greedy2',  filename ):
        return 'Greedy2'
    elif re.search('Relaxed', filename):
        return 'CR'
    else:
        return 'LBSB' 
def whichTopology( filename):
    "Find filename corresponds to which topology."
    for topology in topologies:
        if re.search(topology, filename):
            return topology
    
def whichCongestion( filename):
    for congestion in ['0.95', '0.85', '0.8', '0.7', '0.6', '0.5', '1.0']:
        if re.search(congestion, filename):
            return congestion

def barPlotter(DICS, outfile, y_axis_label = 'Objective', normalize=False):
    def formVals(DICS_alg):
        out = [DICS_alg[key] for key in topologies]
        labels =  topologies
        return out, labels
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 4)
    width = 1


    N = len(DICS[DICS.keys()[0]].keys()) 
    numb_bars = len(DICS.keys() ) + 1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    RECTS = []
    i = 0


    if args.normalize:
        plt.ylim([0,1.1])
        y_axis_label = "Normalized " + y_axis_label
        LOGBAR = False
    else:
        plt.ylim([0.1, 50])
        LOGBAR = True

    for key  in Algorithms:
        values, labels = formVals(DICS[key])
        print values,  key
    #    ax.bar(ind+i*width, values, align='center',width=width, color = colors[i], hatch = hatches[i],label=alg,log=True)
        RECTS+= ax.bar(ind+i*width, values, align='center',width=width, color = colors[i], hatch = hatches[i],label=key,log=LOGBAR)
        i+=1
    if args.lgd:
        LGD = ax.legend(Algorithms, ncol=len(DICS.keys() ), borderaxespad=0.,loc=3, bbox_to_anchor=(0., 1.02, 1., .102),fontsize=15,mode='expand')    
    else:
        LGD = None
    
    print labels 
    ax.set_xticks(ind + width) 
    ax.set_xticklabels(tuple(labels),fontsize = 18)
    ax.set_ylabel(y_axis_label,fontsize=18)
  #  ax.set_yticklabels([0, 0.5, 1])
    plt.yticks(fontsize = 18)
    plt.xlim([ind[0]-width,ind[-1]+len(DICS.keys() )*width])
        
    fig_size = plt.rcParams["figure.figsize"]
    if args.lgd:
       # fig.savefig(outfile+".pdf",format='pdf', bbox_extra_artists=(LGD,), bbox_inches=Bbox(np.array([[0,0],[20,8]])) )
        fig.savefig(outfile+".pdf",format='pdf', bbox_inches='tight')
    else:
        fig.savefig(outfile+".pdf",format='pdf', bbox_inches='tight' )
    plt.show()    

def linePlotter(DICS, outfile, yaxis_label='Objective', xaxis_label='Looseness coefficient $\kappa$', x_scale='linear', y_scale='linear'):
    def formVals(DICS_alg):
        vals = []
        x_axis = []
        for key in sorted( [eval(val) for val in DICS_alg.keys()] ):
            vals.append( DICS_alg[str(key)]  )
            x_axis.append(key )
        return vals, x_axis
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    i=0
    for alg in Algorithms:
        vals, x_axis = formVals(DICS[alg]) 
        plt.plot(x_axis, vals, lin_styles[i], label=alg, linewidth=3, markersize=18)
        i += 1
    plt.xlabel(xaxis_label,fontsize = 18)
    plt.ylabel(yaxis_label, fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 16)

    lgd = plt.legend( loc='upper left',bbox_to_anchor=(0,1),ncol=1,borderaxespad=0.,fontsize= 16)

  #  plt.xlim(0.5,1.1)
    plt.xscale(x_scale)
    plt.yscale(y_scale )


    fig.savefig(outfile+'.pdf',  bbox_extra_artists=(lgd,), format='pdf', bbox_inches='tight' )

def doubleAxeslinePlotter(DIC, outfile, yaxis_label='Objective', y2axis_label='Satisfied Constraints Ratio', xaxis_label='Time (s)', x_scale='linear', y_scale='linear'):
    def formVals(DIC_alg):
        vals = []
        vals2 = []
        x_axis = []
        for key in sorted( [eval(val) for val in DIC_alg.keys()] ):
            print key
            vals.append( DIC_alg[str(key)][0]  )
            vals2.append( DIC_alg[str(key)][1]  )
            x_axis.append(key )
        return vals, vals2, x_axis
    fig, ax = plt.subplots()

    ax2 = ax.twinx()

   # fig.set_size_inches(8, 6)
    i=0
    vals, vals2, x_axis = formVals(DIC)
    ax.plot(x_axis, vals, lin_styles[i], label=yaxis_label, linewidth=3, markevery=1, markersize=18)
    ax2.plot(x_axis, vals2, lin_styles[i+1], label=y2axis_label,  linewidth=3, markevery=1, markersize=18)
    
    ax.set_xlabel(xaxis_label,fontsize = 16)
    ax.set_ylabel(yaxis_label, fontsize = 16)
    ax2.set_ylabel(y2axis_label, fontsize = 16)

    ax.tick_params(axis="y", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)

    plt.xticks(fontsize = 18)

    lgd = ax.legend( loc='upper left',bbox_to_anchor=(0.1, 1),ncol=1,borderaxespad=0.,fontsize= 14)
    lgd2 = ax2.legend( loc='upper left',bbox_to_anchor=(0.1,0.9),ncol=1,borderaxespad=0.,fontsize= 14)
    lgd.get_frame().set_edgecolor('w')
    lgd2.get_frame().set_edgecolor('w')

  #  plt.xlim(0.5,1.1)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale )
    ax2.set_yscale(y_scale)


    fig.savefig(outfile+'.pdf', bbox_extra_artists=(lgd,lgd2), format='pdf', bbox_inches='tight' )
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
    parser.add_argument('--normalize',action='store_true',help='Pass to normalize the plot.')
    parser.add_argument('--lgd',action='store_true',help='Pass to make a legened.')
    parser.add_argument('--metric',choices=['OBJ','time'], default='OBJ',help='Determine whether to plot gain or cost')
    parser.add_argument('--plot_type', choices=['bar', 'line'], default='bar')
    args = parser.parse_args()
    
    
    DICS = {}
    
    max_dict = {}
    for filename in args.filenames:
        Alg  = whichAlg(filename)
        if args.plot_type == 'bar':
            Key = whichTopology(filename)
        else:
            Key = whichCongestion(filename)
       
        with open(filename) as current_file:
            alg_args, trace  = pickle.load(current_file)
        trace_lastKey = max(trace.keys() )
        if Alg not in DICS:
             DICS[Alg] = {}

        try:
            DICS[Alg][Key] =  trace[trace_lastKey ][args.metric]
        except KeyError:
            print filename
    #        continue
         
        if args.metric == 'OBJ':
            y_axis_label = 'Objective'
            DICS[Alg][Key] *= -1
            if args.normalize and Alg == 'LBSB':
                max_dict[Key] = DICS[Alg][Key]
        else:
            y_axis_label = 'Time(s)'

    #Normalize
    if args.normalize:
        for alg in DICS:
            for Key in DICS[alg]:
                print DICS[alg][Key], alg, Key
                DICS[alg][Key] /= max_dict[Key]
                print DICS[alg][Key] 
    if args.plot_type == 'bar':
        barPlotter(DICS, args.outfile, y_axis_label)               
    else:
        linePlotter(DICS, args.outfile, y_axis_label)
    


