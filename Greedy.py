from topologyGenerator import Problem
from helpers import reverseDict,clearFile
import numpy as np
from cvxopt import matrix, spmatrix, solvers
import argparse
import logging

solvers.options['show_progress'] = False

class maximizeUtility():
    def __init__(self,problem_instance):
        self.problem_instance = problem_instance
    def makeContstraintMatrix(self):
        row = 0
        col = 0
        edge2row = {}
        demand2col = {}
        maxRateCol = {}
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
                maxRateCol[col] = maxRate
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

        #Add the non-negativity constraints
        vector_value_list += [0.0] * col
        vector_value_rows += [row + ind for ind in range(col) ]
        matrix_value_list += [-1.0] * col
        matrix_value_rows += [row + ind for ind in range(col)]
        matrix_value_cols += [ind for ind in range(col)] 
        #Add upper bound (maxRate) on the rates
        vec_values_appended = [0.0] * col
        matrix_value_list_appended = []
        matrix_value_cols_appended = []
        for ind in maxRateCol:
             vec_values_appended[ind] = maxRateCol[ind]
             matrix_value_cols_appended.append(ind)
             matrix_value_list_appended.append(1.0)
            
             
        vector_value_list += vec_values_appended
        vector_value_rows += range(row + col, row + 2 * col)
        matrix_value_list += matrix_value_list_appended
        matrix_value_rows += range(row + col, row + 2 * col)
        matrix_value_cols += matrix_value_cols_appended
        

        
        
        
        #Write variable to indices 
        self.maxRateCol = maxRateCol
        self.edge2row = edge2row
        self.demand2col = demand2col
        self.row2edge = reverseDict(edge2row)
        self.col2demand = reverseDict(demand2col) 
        #Make matrices
        total_rows = row + 2 * col 
        total_cols = col
        G = spmatrix(matrix_value_list, matrix_value_rows, matrix_value_cols, (total_rows, total_cols))
        h = spmatrix(vector_value_list,  vector_value_rows, [0 for r in range(total_rows)], (total_rows, 1))
        return G, h
    def maximize(self):
        #Make the constraints 
        G, h = self.makeContstraintMatrix()
        numb_constraints, numb_vars = G.size
        #Make the objective
        def F(x=None, z=None):
            if x is None:
                return (0, matrix(0.0, (numb_vars,1)))

            #Check if x is in the domain of the objective
            if min(x) < -1 * self.problem_instance.log_margin:
                return None
            #Evaluate objective and the grdaients for the given point x
            for elem in range(x.size[0]):
                self.problem_instance.VAR[ self.col2demand[elem] ] = self.maxRateCol[elem] - x[elem]
            obj, grad, Hessian = self.problem_instance.evalGradandUtilities()
   
            
            f = matrix(  sum( obj.values()) )
            Df = matrix(0.0, (1, numb_vars))
            for item_path in grad:
                #Note that the gradient is multiplied with -1 as the gradient is with respect to the rates rather than rate reminders. 
                Df[ self.demand2col[item_path]  ] += -1.0 * grad[item_path][item_path] 
            if z is None:
                return f, Df
            H = matrix(0.0, (numb_vars, numb_vars))
            for item_path in Hessian: 
                ind = self.demand2col[item_path]
                H[ind, ind] =  Hessian[item_path][(item_path, item_path)] * z
            return f, Df, H
                
        opt_sol = solvers.cp(F, G=G, h= matrix(h) )['x']
        #set variables to optimal solution
        for elem in range(numb_vars):
            self.problem_instance.VAR[ self.col2demand[elem] ] = self.maxRateCol[elem] - opt_sol[elem]      
                   
            
            
    
            
               
                
                
                

def greedyCache(problem_instance):
    "Find an item and a node, s.t., caching the item on that node has the maximum decrease in total flow over the network." 
    deltas = {}
    
    cache_constraints = problem_instance.evalGradandCapcityConstraints(0)[0]
    for var in problem_instance.VAR:
        if type(var[1]) == tuple:
            continue 
        item, node = var
        if problem_instance.VAR[(item, node)] < 1.0 and cache_constraints[node] > 0.0:
            delta_item_node_pair = problem_instance.evalDelta(item, node)
            deltas[(item, node)] = delta_item_node_pair
    if len(deltas.items()) == 0:
       return False
    max_pair, max_delta = max(deltas.items(), key=lambda keyVal: keyVal[1])
    item, node = max_pair
    problem_instance.VAR[(item, node)] = 1.0
    return True 
def continiousGreedyCache(problem_instance, iterations=100):
    "Minimize total flow in othe network, While rates are fixed. Note that the objective is a super-modular function."

    def getMax(grad, cap):
        allElems = dict( [(node,{}) for node in cap])
        chosenElems = []
        allItems = []
        for (item, node) in grad:
            allElems[node][item] = grad[(item, node)]
            if item not in allItems:
                allItems.append(item)
        for node in allElems:
            dict_node = sorted(allElems[node].items(), key=lambda key_val:-1.0 * key_val[1] )
                       
            for chosen_item in dict_node[:cap[node]]:
                item, grad_item  = chosen_item
                chosenElems.append((item, node))
        return chosenElems    
              
        
  
                    
            
    gamma = 1.0/iterations
    t = 0.0
    while t<1.0:
        current_grad = {}
        current_obj = 0.0
        edge_functions, edge_grads, DUMMY_edge_Hessian = problem_instance.evalGradandConstraints(degree=1)
        for edge in edge_grads:
            for var in edge_grads[edge]:
                #Rate reminders are fixed. Only consider the gradient w.r.t. caching variables.
                if type(var[1]) == tuple:
                    continue
                if var not in current_grad:
                    current_grad[var] = edge_grads[edge][var]
                else:
                    current_grad[var] += edge_grads[edge][var] 
        current_subProblem_opt = getMax(current_grad, problem_instance.capacities) 

        #set setp-size
        step_size = min(1.0-t, gamma)
        #update solution
        for var in current_subProblem_opt:
            problem_instance.VAR[var] += step_size 
        t += step_size 

class Greedy1():
    def __init__(self, problem_instance, logger=None):
        self.problem_instance = problem_instance
        self.logger = logger
       
    def optimize(self):
        #Set caching (all) variables to zero
        self.problem_instance.setVAR2Zero()
    
        OBJ, CONSTRAINT = self.problem_instance.evaluate()
        self.logger.info( "Objective is %.3f the total link constraints sum is %.3f" %(OBJ, CONSTRAINT))
        #Determine rates by maximizng the total utility w.r.t. rates
        MU = maximizeUtility(self.problem_instance)
        MU.maximize()
        OBJ, CONSTRAINT = self.problem_instance.evaluate()
        self.logger.info( "Objective is %.3f the total link constraints sum is %.3f" %(OBJ, CONSTRAINT))
        #Cache items
        continiousGreedyCache(self.problem_instance) 
        OBJ, CONSTRAINT = self.problem_instance.evaluate()
        self.logger.info( "Objective is %.3f the total link constraints sum is %.3f" %(OBJ, CONSTRAINT))
        #Determine rates by maximizng the total utility w.r.t. rates
       # MU = maximizeUtility(problem_instance)
        MU.maximize()
        OBJ, CONSTRAINT = self.problem_instance.evaluate()
        self.logger.info( "Objective is %.3f the total link constraints sum is %.3f" %(OBJ, CONSTRAINT))
class Greedy2(Greedy1):
    def optimize(self):
        #Set caching (all) variables to zero
        self.problem_instance.setVAR2Zero()
        MU = maximizeUtility(problem_instance) 
        while greedyCache( self.problem_instance):
            MU.maximize()
            OBJ, CONSTRAINT = self.problem_instance.evaluate()
            self.logger.info( "Objective is %.3f the total link constraints sum is %.3f" %(OBJ, CONSTRAINT))
        
    
   
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Run the Shifted Barrier Method for  Optimizing Network of Caches',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('problem',help = 'Caching problem instance filename')
    parser.add_argument('opt_problem',help = 'Optimized caching problem instance filename')
    parser.add_argument('method', help='Which greedy algrotihm to use', choices=['Greedy1', 'Greedy2'])
    parser.add_argument('--innerIterations',default=10,type=int, help='Number of inner iterations')
    parser.add_argument('--outerIterations',default=1,type=int, help='Number of outer iterations')
    parser.add_argument('--logfile',default='logfile',type=str, help='logfile')
    parser.add_argument('--debug_level',default='INFO',type=str,help='Debug Level', choices=['INFO','DEBUG','WARNING','ERROR'])
    parser.add_argument('--logLevel',default='INFO', help='Verbosity level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    args = parser.parse_args()


    logger = logging.getLogger('Greedy Method')
    logger.setLevel(eval("logging."+args.logLevel))
    clearFile(args.logfile)
    fh = logging.FileHandler(args.logfile)
    fh.setLevel(eval("logging."+args.logLevel))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)


    problem_instance = Problem.unpickle_cls(args.problem)

    greedyClass = eval(args.method)
    GD = greedyClass(problem_instance, logger)
    GD.optimize()
    print "Variables are: ", problem_instance.VAR
    problem_instance.pickle_cls( args.opt_problem )

