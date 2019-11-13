from helpers import Project2Box, ProjOperator, squaredNorm
import random
from topologyGenerator import Problem 
import numpy as np
import argparse


class BarrierOptimizer():
    def __init__(self, Pr):
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
        for constraint in constraint_func:
            if constraint_func[constraint] > 0:
                self.LAMBDAS[constraint] = 1.e-1
            else:
                self.LAMBDAS[constraint] = (-1.0 * constraint_func[constraint] / self.MU) ** (1./self.alpha_lambda) + 1.e-1

        self.LAMBDA_BAR = self.LAMBDAS
           


    def PGD(self, Pr, iterations=100):
        #The follwowing dictionaries keep track of the gradient of the barrier function`s objective 
        self.grad_Psi_VAR  = dict( [(key, 0.0) for key in Pr.VAR] )
        self.grad_Psi_REM  = dict( [(key, 0.0) for key in Pr.REM] )
        for t in range(iterations):
            #* step_size can be computed in other ways 
            step_size = 1./(t+2)

            #Reset the gradientes to zero
            for key in self.grad_Psi_VAR:
                self.grad_Psi_VAR[key] = 0.0
            for key in self.grad_Psi_REM:
                self.grad_Psi_REM[key] = 0.0
            OLDVAR = Pr.VAR
            OLDREM = Pr.REM
             
            #Gradient descent step 
            #w.r.t. constraints
            constraint_grads, constraint_func = Pr.evalFullConstraintsGrad()
       
            for constraint in constraint_grads:
                self.LAMBDA_BAR[constraint] =  self.LAMBDAS[constraint] * self.SHIFTS[constraint] / (constraint_func[constraint] + self.SHIFTS[constraint])
                for index in constraint_grads[constraint]:
                    grad_index = -1.0 * self.LAMBDA_BAR[constraint] * constraint_grads[constraint][index] 


                    # find whether index is a cahing variable or a utility remainder
                    try:
                        Pr.VAR[index] -= step_size * grad_index
                        self.grad_Psi_VAR[index] += grad_index 
                    except KeyError: 
                        print index, grad_index
                        Pr.REM[index] -= step_size * grad_index
                        self.grad_Psi_REM[index] += grad_index
            #w.r.t. objective 
            obj_grads, obj_func = Pr.evalGradandUtilities() 
            for index in obj_grads:
                print index, grad_index
                Pr.REM[index] -= step_size * obj_grads[index]
                self.grad_Psi_REM[index] += obj_grads[index]

            print Pr.REM
            #Projections 
            Project2Box(Pr.VAR, dict( [(var, 1) for var in Pr.VAR] ) )
            Project2Box(Pr.REM, Pr.MAXRATE)

            #Report stats and objective
            OldObj =  sum( obj_func.values() ) 
            print Pr.REM
            for  constraint in constraint_func:
                OldObj -=  self.LAMBDAS[constraint] * self.SHIFTS[constraint] * np.log(constraint_func[constraint] + self.SHIFTS[constraint])
            print("ITERATION %d, current objective value is %f" %(t, OldObj))

            #Optimiality
            non_optimality_VAR = ProjOperator(OLDVAR, self.grad_Psi_VAR, dict( [(var, 1) for var in Pr.VAR] ))
            non_optimality_REM = ProjOperator(OLDREM, self.grad_Psi_REM, Pr.MAXRATE) 
            non_optimality_norm = np.sqrt(squaredNorm(non_optimality_VAR)**2 + squaredNorm(non_optimality_REM)**2)
            print("ITERATION %d, current non-optimality is %f" %(t, non_optimality_norm)) 
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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Run the Shifted Barrier Method for  Optimizing Network of Caches',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('problem',help = 'Caching problem instance filename')
    parser.add_argument('--iterations',default=100,type=int, help='Number of iterations') 
    args = parser.parse_args()

    
    problem_instance = Problem.unpickle_cls(args.problem) 
    ##Debugging

    #The gradient computation 
    eps = 1.e-3
    #Eval grads and functions 
    constraint_grads, constraint_func = problem_instance.evalFullConstraintsGrad()
    for i in range(0):
        #Pick random constraints and variables
        const = random.choice( constraint_grads.keys() )
        index  = random.choice( constraint_grads[const].keys() )
        print "The computed gradient of constraint ",const, " w.r.t. ", index, " is ", constraint_grads[const][index] 
        try:
            problem_instance.VAR[index] += eps
        except KeyError:
            problem_instance.REM[index] += eps

        constraint_grads_eps, constraint_func_eps = problem_instance.evalFullConstraintsGrad()
        print "The emperical gradient of constraint ", const,  " w.r.t. ", index, " is ", (constraint_func_eps[const] - constraint_func[const])/eps

        try:
            problem_instance.VAR[index] -= eps
        except KeyError:
            problem_instance.REM[index] -= eps
       
    optimizer = BarrierOptimizer(problem_instance)
    optimizer.outerIter(problem_instance, 1, 100)
    
    
        
        
                


        
        
