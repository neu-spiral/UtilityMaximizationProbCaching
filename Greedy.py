from topologyGenerator import Problem
from cvxopt import matrix, spmatrix
import argparse


class maximizeUtility():
    def __init__(self,problem_instance):
        self.problem_instance = problem_instance
    def _makeContstraintMatrix(self):
        row = 0
        col = 0
        edge2row = {}
        demand2col = {}

        matrix_value_list = []
        matrix_value_rows = []
        matrix_value_cols = []
        vector_value_list = []
        vector_value_rows = []
        
        for demand in self.problem_instance.demands:
            item = demand['item']
            maxRate = demand['rate']
            path = tuple(demand['path'])
            curr_prod = 1.0
            if (item, path) not in demand2col:
                demand2col[(item, path)] = col
                col += 1  
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])

                if edge not in edge2row:
                    edge2row[edge] = row
                    row += 1
                vector_value_list.append(    self.problem_instance.bandwidths[edge] )
                vector_value_rows.append(  edge2row[edge] ) 
  
                curr_prod *= (1.0 - self.problem_instance.VAR[(item, path[i])] )
                matrix_value_list.append( curr_prod )
                matrix_value_rows.append(  edge2row[edge] )
                matrix_value_cols.append(  demand2col[(item, path)] )

        #Write variable to indices 
        self.edge2row = edge2row
        self.demand2col = demand2col
        #Make matrices
        G = spmatrix(matrix_value_list, matrix_value_rows, matrix_value_cols, (row, col))
        b = spmatrix(vector_value_list,  vector_value_rows, [0 for r in vector_value_rows], (row, 1))
        print b
        return G, b
            
               
                
                
                

def greedyCache(problem_instance):
    "Given a problem instance including caching and rate reminder variables choose an item and a node to set to one."
    deltas = {}
    for var in problem_instance.VAR:
        if type(var[1]) == tuple:
            continue 
        item, node = var
        if problem_instance.VAR[(item, node)] < 1.0:
            delta_item_node_pair = problem_instance.evalDelta(item, node)
            deltas[(item, node)] = delta_item_node_pair
            print "Delta for", item, node, " is ", delta_item_node_pair
    max_pair, max_delta = max(deltas.items(), key=lambda keyVal: keyVal[1])
    item, node = max_pair
    problem_instance.VAR[(item, node)] = 1.0


def Greedy(problem_instance):
    pass


if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Run the Shifted Barrier Method for  Optimizing Network of Caches',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('problem',help = 'Caching problem instance filename')
    parser.add_argument('opt_problem',help = 'Optimized caching problem instance filename')
    parser.add_argument('--innerIterations',default=10,type=int, help='Number of inner iterations')
    parser.add_argument('--outerIterations',default=1,type=int, help='Number of outer iterations')
    parser.add_argument('--logfile',default='logfile',type=str, help='logfile')
    parser.add_argument('--debug_level',default='INFO',type=str,help='Debug Level', choices=['INFO','DEBUG','WARNING','ERROR'])
    parser.add_argument('--logLevel',default='INFO', help='Verbosity level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    args = parser.parse_args()


    problem_instance = Problem.unpickle_cls(args.problem)

    problem_instance.setVAR2Zero()
    print "Demands: ", problem_instance.demands
    MU = maximizeUtility(problem_instance)
    print MU._makeContstraintMatrix()
    eps = 1.e-3

