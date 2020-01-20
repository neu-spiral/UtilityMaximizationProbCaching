from helpers import Project2Box, ProjOperator, squaredNorm, clearFile, Proj2PositiveOrthant
import logging 
import random
import math 
import sys
from topologyGenerator import RelaxedProblem
import numpy as np
import argparse
from SparseVector import SparseVector
import time
import pickle


class SubGradOptimizer:
    def __init__(self, RlaxedPr, logger):
        self.Pr = RlaxedPr
        #Create dual variables LAMBDAS
        constraint_func,constraint_grads, constraint_HESSIAN = self.Pr.evalFullConstraintsGrad(0 )
        self.LAMBDAS = SparseVector( dict([(key, 0.0) for key in constraint_func]  ) )
        #step-size parameter
        self.beta = 2.0
        self.gamma = 1.0
        self.logger = logger

    def initialPoint(self):
        for key in self.Pr.VAR:
            self.Pr.VAR[key] = 0.0
    def evaluate(self, degree=1, debug=False):
        """Evalue the Lagrangian function. 
           degree = 0 computes the Lagrangian 
           degree = +1 computes the Lagrnagian plus its gradient
        """
        obj_lagrangian = 0.0
        grad_lagrangian  = {}
        #w.r.t. constraints
        constraint_func, constraint_grads, constraint_Hessian = self.Pr.evalFullConstraintsGrad(degree)
        #w.r.t. objective 
        obj_func, obj_grads, obj_Hessian = self.Pr.evalGradandUtilities(degree)
      

        for obj in obj_func:
            #Objective
            obj_lagrangian += obj_func[obj]

            if degree<1:
                continue
            #Grad
            for index in obj_grads[obj]:
                if index in grad_lagrangian:
                    grad_lagrangian[index] += obj_grads[obj][index]
                else:
                    grad_lagrangian[index] = obj_grads[obj][index]

        utility = obj_lagrangian

        for constraint in constraint_func:
            #Objective
            obj_lagrangian += -1.0 * self.LAMBDAS[constraint] * constraint_func[constraint] 
            if degree<1:
                continue
            #Grad
            for index in constraint_grads[constraint]:
                grad_index = -1.0 * self.LAMBDAS[constraint] * constraint_grads[constraint][index]
                if index in grad_lagrangian:
                    grad_lagrangian[index] += grad_index
                else:
                    grad_lagrangian[index] = grad_index
        
        dual_grad = SparseVector( constraint_func ) * -1.0        
        feasibility = 1.0 * sum( [dual_grad[key] <= 0.0 for key in dual_grad   ]) /  len(dual_grad.keys())
        return obj_lagrangian, SparseVector(grad_lagrangian), dual_grad, utility, feasibility
        
    
        
    def Primal(self, iterations=100, eps=1.e-3):
        obj_lagrangian, grad_lagrangian, dual_grad, utility, feasibility = self.evaluate()
        OBJ_k = obj_lagrangian
        VAR_k = self.Pr.VAR
        for t in range(iterations):
            #Grdianet desecnet 
            self.Pr.VAR -= 1./(t+2) * grad_lagrangian
            #Projection
            Project2Box(self.Pr.VAR, self.Pr.BOX )
            #Evaluate objective and gradient 
            obj_lagrangian, grad_lagrangian, dual_grad, utility, feasibility = self.evaluate()
            OBJ_k = min(OBJ_k, obj_lagrangian)
            if obj_lagrangian <= OBJ_k:
                VAR_k = self.Pr.VAR
            if t % 50 ==0:
                self.logger.info("Primal iteration %d, best objectve so far is %.3f, objective is %.3f and %.3f." %(t+1, OBJ_k, obj_lagrangian, feasibility))
            
        self.Pr.VAR = VAR_k
        obj_lagrangian, grad_lagrangian, dual_grad, utility, feasibility = self.evaluate()
        return dual_grad, obj_lagrangian, utility, feasibility
          
    def solve(self, iterations=100, inner_iterations=100):
        #Evaluate the dual upper bound
        for var in self.Pr.VAR:
            self.Pr.VAR[var] = 1.0
        obj_lagrangian, grad_lagrangian, dual_grad, upper_bound, feasibility = self.evaluate()
        #set initial point to a feasible primal point
        self.initialPoint()
        trace = {}
        t_start = time.time()
        obj_lagrangian, grad_lagrangian, dual_grad, primal_obj_feasible, feasibility = self.evaluate()
        
        for t in range(iterations):
            trace[t] = {}
            dual_grad, dual_obj, primal_obj, primal_feasibility = self.Primal(iterations = inner_iterations)
            if t == 0:
                max_dual = dual_obj
            else:
                max_dual = max(max_dual, dual_obj)
            q_opt_estimate = .5 * (max_dual + upper_bound)
            #Set step-size
            alpha_t = self.beta/(self.gamma + t)
            step_size = alpha_t * (q_opt_estimate - dual_obj) / ( squaredNorm(dual_grad) ) ** 2
            #Aadapt dual variables
            self.LAMBDAS += step_size * dual_grad 
            
            Proj2PositiveOrthant(self.LAMBDAS)

            now = time.time()
            
            trace[t]['DUALOBJ'] = max_dual
            trace[t]['OBJ'] = primal_obj
            trace[t]['time'] = now - t_start
            trace[t]['RelaxedCONSTRAINT'] = primal_feasibility
            trace[t]['CONSTRAINT'] = self.Pr.evalOriginalConstraints()
            self.logger.info("Iteration %d, primal objective is %.4f, dual objective is %.4f, constraint satisfation %.4f, original constraint satisfation %.4f" %(t, primal_obj, dual_obj,trace[t]['RelaxedCONSTRAINT'] , trace[t]['CONSTRAINT']))

        return trace 
             
          
            
        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Run the Shifted Barrier Method for  Optimizing Network of Caches',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('problem',help = 'Caching problem instance filename')
    parser.add_argument('opt_problem',help = 'Optimized caching problem instance filename')
    parser.add_argument('trace_file',help = 'Trace file')
    parser.add_argument('--innerIterations',default=10,type=int, help='Number of inner iterations') 
    parser.add_argument('--outerIterations',default=1,type=int, help='Number of outer iterations')
    parser.add_argument('--logfile',default='logfile',type=str, help='logfile')
    parser.add_argument('--debug_level',default='INFO',type=str,help='Debug Level', choices=['INFO','DEBUG','WARNING','ERROR'])
    parser.add_argument('--logLevel',default='INFO', help='Verbosity level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    args = parser.parse_args()


    
    problem_instance = RelaxedProblem.unpickle_cls(args.problem) 
    eps = 1.e-3
       
    #set up logfile
    logger = logging.getLogger('Sub-gradient Method')
    logger.setLevel(eval("logging."+args.logLevel)) 
    clearFile(args.logfile)
    fh = logging.FileHandler(args.logfile)
    fh.setLevel(eval("logging."+args.logLevel))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
   

    optimizer = SubGradOptimizer(problem_instance, logger)
    trace = optimizer.solve( args.outerIterations, args.innerIterations )
    problem_instance.pickle_cls( args.opt_problem )
    
    with open(args.trace_file,'wb') as f:
        pickle.dump((args,trace),f)
    
    
        
        
                


        
        
