from helpers import Project2Box, ProjOperator, squaredNorm, clearFile
import logging 
import random
from topologyGenerator import Problem 
import numpy as np
import argparse


class BarrierOptimizer():
    def __init__(self, Pr, logger=None):
        #initialize algorithm parameters 
        initVal = 0.5
        self.eta_s = initVal
        self.omega_s = initVal
        self.alpha_omega = initVal
        self.beta_omega = initVal
        self.beta_eta = initVal
        self.alpha_lambda = initVal
        self.tau = 0.5
        self.rho = 0.5
        self.omega_star = 1.e-4
        self.eta_star = 1.e-4
        self.alpha_eta = 1.1 - 1./(1+ self.alpha_lambda)
        self.MU = initVal
        self.OMEGA  = self.omega_s * self.MU**self.alpha_omega 
        self.ETA = self.eta_s * self.MU ** self.alpha_eta
        #Initialize the dual variables LAMBDAS
        constraint_grads, constraint_func = Pr.evalFullConstraintsGrad()
        self.LAMBDAS = {}
        #* eps
        eps_margin = 1.e-1
        for constraint in constraint_func:
            if constraint_func[constraint] > 0:
                self.LAMBDAS[constraint] =  (eps_margin / self.MU) ** (1./self.alpha_lambda) 
            else:
                self.LAMBDAS[constraint] = ( (-1.0 * constraint_func[constraint] + eps_margin) / self.MU) ** (1./self.alpha_lambda) 

        self.LAMBDA_BAR = self.LAMBDAS
        #logger
        self.logger = logger
           


    def PGD(self, Pr, iterations=100):
        #The follwowing dictionaries keep track of the gradient of the barrier function`s objective 
        self.grad_Psi_VAR  = {}
        for t in range(iterations):
            #* step_size can be computed in other ways 
            step_size = 1./(t+2)

            #Set/Reset the gradientes to zero
            for key in Pr.VAR:
                self.grad_Psi_VAR[key] = 0.0
             
            #Gradient descent step 
            #w.r.t. constraints
            constraint_grads, constraint_func = Pr.evalFullConstraintsGrad()
            #w.r.t. objective 
            obj_grads, obj_func = Pr.evalGradandUtilities()
       
            for constraint in constraint_grads:
                self.LAMBDA_BAR[constraint] =  self.LAMBDAS[constraint] * self.SHIFTS[constraint] / (constraint_func[constraint] + self.SHIFTS[constraint])
                for index in constraint_grads[constraint]:
                    grad_index = -1.0 * self.LAMBDA_BAR[constraint] * constraint_grads[constraint][index] 
                    Pr.VAR[index] -= step_size * grad_index
                    self.grad_Psi_VAR[index] += grad_index 

            for index in obj_grads:
                Pr.VAR[index] -= step_size * obj_grads[index]
                self.grad_Psi_VAR[index] += obj_grads[index]

            #Projections 
            Project2Box(Pr.VAR, Pr.BOX)

            #Report stats and objective
            OldObj =  sum( obj_func.values() ) 
            for  constraint in constraint_func:
                OldObj -=  self.LAMBDAS[constraint] * self.SHIFTS[constraint] * np.log(constraint_func[constraint] + self.SHIFTS[constraint])
            self.logger.info("INNER ITERATION %d, current objective value is %f" %(t, OldObj))

            #Optimiality
            non_optimality_VAR = ProjOperator(Pr.VAR, self.grad_Psi_VAR, Pr.BOX)
            non_optimality_norm = squaredNorm( non_optimality_VAR ) 
            self.logger.info("INNER ITERATION %d, current non-optimality is %f" %(t, non_optimality_norm)) 
            if non_optimality_norm<self.OMEGA:
                break
        return constraint_func, non_optimality_norm   
        
    def outerIter(self, Pr, OuterIterations, InnerIterations):
       
        for k in range(OuterIterations):
            self.SHIFTS  = dict([(edge, self.MU * (self.LAMBDAS[edge] ** self.alpha_lambda)) for edge in self.LAMBDAS] )
            #Inner iteration
            constraint_func, non_optimality_norm = self.PGD(Pr, iterations = InnerIterations)
        
            comp_slack  = dict([(key, self.LAMBDA_BAR[key] * constraint_func[key]) for key in self.LAMBDA_BAR])

            comp_slack_norm = squaredNorm(comp_slack)
    
            if comp_slack_norm < self.eta_star and non_optimality_norm < self.omega_star:
                 break
            if squaredNorm( dict([(key, comp_slack[key]/(self.LAMBDAS[key]**self.alpha_lambda)) for key in self.LAMBDAS] ) ) <= self.ETA:
                #Exe. Step 3
                self.LAMBDAS = self.LAMBDA_BAR
                self.OMEGA *= (self.MU ** self.beta_omega)
                self.ETA *= (self.MU ** self.beta_eta) 
            else:
                #Exe. Step 4
                self.MU *= self.tau
                self.OMEGA = self.omega_s * self.MU ** self.alpha_omega
                self.ETA = self.eta_s * self.MU ** self.alpha_eta

            self.logger.info("OUTER ITERATION %d, current non-optimality is %f, current complimentary slackness violation is %f" %(k, non_optimality_norm, comp_slack_norm) )
            

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Run the Shifted Barrier Method for  Optimizing Network of Caches',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('problem',help = 'Caching problem instance filename')
    parser.add_argument('--iterations',default=100,type=int, help='Number of iterations') 
    parser.add_argument('--logfile',default='logfile',type=str, help='logfile')
    parser.add_argument('--logLevel',default='INFO', help='Verbosity level',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    args = parser.parse_args()

    
    problem_instance = Problem.unpickle_cls(args.problem) 
    ##Debugging


    #Random variables for vraiables
    for key in problem_instance.VAR:
         problem_instance.VAR[key] = 1.0
    #The gradient computation 
    eps = 1.e-3
    #Eval grads and functions 
    constraint_grads, constraint_func = problem_instance.evalFullConstraintsGrad()
    for i in range(0):
        #Pick random constraints and variables
        const = random.choice( constraint_grads.keys() )
        index  = random.choice( constraint_grads[const].keys() )
        print "The computed gradient of constraint ",const, " w.r.t. ", index, " is ", constraint_grads[const][index] 
        problem_instance.VAR[index] += eps

        constraint_grads_eps, constraint_func_eps = problem_instance.evalFullConstraintsGrad()
        print "The emperical gradient of constraint ", const,  " w.r.t. ", index, " is ", (constraint_func_eps[const] - constraint_func[const])/eps

        problem_instance.VAR[index] -= eps
       
    #set up logfile
    logger = logging.getLogger('Shifted Barrier Method')
    logger.setLevel(eval("logging."+args.logLevel)) 
    clearFile(args.logfile)
    fh = logging.FileHandler(args.logfile)
    fh.setLevel(eval("logging."+args.logLevel))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    

    optimizer = BarrierOptimizer(problem_instance, logger)
    optimizer.outerIter(problem_instance, 10, 100)
    
    
        
        
                


        
        
