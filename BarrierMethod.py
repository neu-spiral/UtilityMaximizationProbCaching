from helpers import Project2Box, ProjOperator, squaredNorm, clearFile
import logging 
import random
import math 
import sys
from topologyGenerator import Problem
import numpy as np
import argparse
from SparseVector import SparseVector
import time
import pickle

class boxOptimizer():
    def __init__(self):
        """This class is the implmentation of the algrotihm prposed in GLOBAL CONVERGENCE OF A CLASS OF
        TRUST REGION ALGORITHMS FOR OPTIMIZATION WITH SIMPLE BOUNDS. It solves generic box constrianed problems of the form
            Minimize F(x)
            Subject to 0 <= x <= B.
        """
        self.mu = 0.5
        self.eta  = 0.6
        self.gamma0 = 0.3
        self.gamma1  = 0.7
        #self.gamma2  = 2.0
        self.gamma2  = 10.0
        self.nu = 1.0
        self.Delta = 0.5
    def initialPoint(self, Pr, SHIFTS, startFromLast=False):
        if startFromLast:
            return 

        constraint_func, constraint_grads_dummy, constraint_Hessian_dummy = Pr.evalFullConstraintsGrad(0)
        FEAS = True
        #If the current 
        for constraint in constraint_func:
           FEAS = FEAS and constraint_func[constraint]  + SHIFTS[constraint] >= 0.0 
        if FEAS:
            return 
        for key in Pr.VAR:
            if type(key[1]) == tuple:
                 #Remainder variables 
                Pr.VAR[key] = Pr.BOX[key]
            else:
                 #Caching variables 
                Pr.VAR[key] = 0.0


    def evaluate(self, Pr, LAMBDAS, SHIFTS, degree=2, debug=False):
        """Evalue the objective function. 
           degree = -1 only computes the LAMBAD BAR
           degree = 0 computes LAMBAD BAR plus the objective value
           degree = +1 computes LAMBDA BAR, the objective, and the objective`s gradient
           degree = +2 computes  LAMBDA BAR, the objective, the objective`s gradient, and the objective`s Hessian  
        """
        obj_barrier = 0.0
        grad_barrier  = {}
        Hessian_barrier = {}
        LAMBDA_BAR = {}
        #w.r.t. constraints
        constraint_func, constraint_grads, constraint_Hessian = Pr.evalFullConstraintsGrad(degree)
        #w.r.t. objective 
        obj_func, obj_grads, obj_Hessian = Pr.evalGradandUtilities(degree)



        for obj in obj_func:
            if degree < 0:
                continue
            #Objective
            obj_barrier += obj_func[obj]

            if degree<1:
                continue
            #Grad
            for index in obj_grads[obj]:
                if index in grad_barrier:
                    grad_barrier[index] += obj_grads[obj][index]
                else:
                    grad_barrier[index] = obj_grads[obj][index]
            if degree<2:
                continue
            #Hessian
            for index_pair in obj_Hessian[obj]:
                if index_pair in Hessian_barrier:
                    Hessian_barrier[index_pair] += obj_Hessian[obj][index_pair]
                else:
                    Hessian_barrier[index_pair] = obj_Hessian[obj][index_pair]



        for constraint in constraint_func:
            LAMBDA_BAR[constraint] =  LAMBDAS[constraint] * SHIFTS[constraint] / (constraint_func[constraint] + SHIFTS[constraint])
            if degree < 0:
                continue
            #Objective
            try:
                obj_barrier += -1.0 * LAMBDAS[constraint] * SHIFTS[constraint] * math.log(constraint_func[constraint] + SHIFTS[constraint])
            except ValueError:
                obj_barrier = float("inf")
            if degree<1:
                continue
            #Grad
            for index in constraint_grads[constraint]:
                grad_index = -1.0 * LAMBDA_BAR[constraint] * constraint_grads[constraint][index]
                if index in grad_barrier:
                    grad_barrier[index] += grad_index
                else:
                    grad_barrier[index] = grad_index

            if degree<2:
                continue
            #Hessian
            for index_pair in constraint_Hessian[constraint]:
                if index_pair in Hessian_barrier:
                     Hessian_barrier[index_pair] += constraint_Hessian[constraint][index_pair]
                else:
                     Hessian_barrier[index_pair] = constraint_Hessian[constraint][index_pair]

        return LAMBDA_BAR, obj_barrier, SparseVector(grad_barrier), SparseVector(Hessian_barrier)
    def optimizer(self, Pr, Lambdas, Shifts, FirstOrderOptThreshold , iterations=100, debug=False, logger=None):
        
        REJ = False
        LAMBDA_BAR, obj, grad, Hessian = self.evaluate(Pr, Lambdas, Shifts)
        #Initialize delta 
        self.Delta = 0.5
        for i in range(iterations):
            TrustRegionThreshold = self.Delta * self.nu   


            #Find a direction for update
            s_k = self._findCauchyPoint(grad, Hessian, Pr.VAR, Pr.BOX, TrustRegionThreshold, debug=False)
            
            #Update the current solution 
            Pr.VAR += s_k


            #Evaluet only the objective for the new point
            LAMBDA_BAR, obj_toBeTested, grad_NULL, Hessian_NULL = self.evaluate(Pr, Lambdas, Shifts, 0) 
            #Measure the improvement raio 
            if  -1.0 * s_k.dot(grad) - 0.5 * s_k.dot( s_k.MatMul(Hessian) )  != 0:
                rho_k = (obj - obj_toBeTested) / (-1.0 * s_k.dot(grad) - 0.5 * s_k.dot( s_k.MatMul(Hessian)  ) ) 
            else:
                #If in the current interval local min is t = 0, make the trust region larger so that the algorithm can find a better local min.
                rho_k =  self.eta
                
             
            if rho_k <= self.mu:
               #Point rejected
                REJ = True
                Pr.VAR -= s_k
                self.Delta *= 0.5 * (self.gamma0 + self.gamma1)  
            elif rho_k < self.eta:
                 #Point accepted
                 REJ = False
                 self.Delta *=  0.5 * (self.gamma1 + 1.0) 
            else:
                 #Point accepted
                 REJ = False
                 self.Delta *=  0.5 * (1.0 + self.gamma2)  
            if not REJ:
                LAMBDA_BAR, obj, grad, Hessian = self.evaluate(Pr, Lambdas, Shifts)
           
            if debug:
                print "Iteration ", i
                print "Thershold ", TrustRegionThreshold
                print "Grad norm", squaredNorm(grad), " accpted: ", not REJ
                print "Direction norm ", squaredNorm( s_k)
                print "Obj improvement is, ", obj - obj_toBeTested
                print "Quad obj improevment is  ",  -1.0 * s_k.dot(grad) - 0.5 * s_k.dot( s_k.MatMul(Hessian) )
                print "Improvement ratio", rho_k
                
          
            #     print "s_k is ", s_k
            #    print "Var is ", Pr.VAR
            #    print "grad is ", grad

            #Stppping criterion 
            
            firstOrderOpt = squaredNorm( ProjOperator(Pr.VAR, grad, Pr.BOX) )
            if i % 50 == 0:
                logger.info("Inner iteration %d, current obejctive value is %.8f and current optimality gap is %.10f" %(i, obj, firstOrderOpt  )  )

           # print "Direction is ", s_k, " rejection ", REJ, " ratio ", rho_k, " variables ", Pr.VAR
            if firstOrderOpt <= FirstOrderOptThreshold:
               break
            
        LAMBDA_BAR, obj_toBeTested, grad_NULL, Hessian_NULL = self.evaluate(Pr, Lambdas, Shifts, 0)
        return LAMBDA_BAR, firstOrderOpt
            
        #m = 
        #rho = (f(x_k) - f(x_k+s_k)) / (f(x_k) - )
        # if rho > mu
        
        #else 

        
    def _getQudraticQuoeff(self, S_independant_k, S_dependant_k, grad, Hessian):
        b = S_dependant_k.dot(grad) + S_dependant_k.dot( S_independant_k.MatMul( Hessian)  )
        a = 0.5 * S_dependant_k.dot( S_dependant_k.MatMul( Hessian)  ) 
        return a, b
      
    def _findCauchyPoint(self, grad, Hessian, Vars, Box, TrustRegionThreshold, scaling=False, debug=False):
        "Return the direction s_k as in Step 1 of the algorithm. Note that grad and Hessian are SparseVectors."
        

        if not scaling:
            scalingD = dict( [(key, 1.0) for key in Vars] )
        else:
            scalingD = {}
            for key in Vars:
                try: 
                    if grad[key] >= 0: 
                        scalingD[key] =  Vars[key]
                    else:
                        scalingD[key] =    (Box[key] - Vars[key] )
                except ZeroDivisionError:
                    scalingD[key] = 10.0
       #Compute hitting times
        hitting_times = {'dummy':0.0 }
        for key in Vars:
            if grad[key] == 0.0:
               hitting_times[key] = sys.maxsize
            elif grad[key] > 0:
                hitting_times[key] = Vars[key] / (scalingD[key] * grad[key])
            else:
                hitting_times[key] = (Box[key] - Vars[key]) / abs(  scalingD[key] * grad[key] )
        
        sorted_hitting_times_items = sorted(hitting_times.items(), key = lambda x: x[1])
        
       #Decompose S_k = S_independant_k + S_dependant_k * t
        S_independant_k = SparseVector({})
        S_dependant_k = SparseVector( dict([(key, -1.0 * scalingD[key] * grad[key]) for key in grad]) )
        vars_indepnedant_of_t = []
        end_deriavative_sgn = 0
        t_threshold = sys.maxsize
        for i in range(len( sorted_hitting_times_items )):
            key, t_key = sorted_hitting_times_items[i]
                 
            if i < len( sorted_hitting_times_items ) -1:
                next_key, next_t_key = sorted_hitting_times_items[i+1]
            else:
                next_t_key = -1 #dummy value 
            if key != 'dummy':
                vars_indepnedant_of_t.append( key )
                del S_dependant_k[key]
                if grad[key] > 0.0:
                    S_independant_k[key] = -1.0 * Vars[key]
                elif grad[key] < 0.0:
                    S_independant_k[key] = Box[key] - Vars[key]
                else:
                    S_independant_k[key] = 0.0
            if next_t_key == t_key:
                continue 
            
            if debug:
                print "Search ointerval is :", t_key, next_t_key
            #for key in vars_indepnedant_of_t:
           #     del S_dependant_k[key]
            a, b = self._getQudraticQuoeff(S_independant_k, S_dependant_k, grad, Hessian)
 
            #Check if the current interval is inside the trusts region
            
            if squaredNorm( S_independant_k + S_dependant_k * next_t_key ) >= TrustRegionThreshold:
                A = S_dependant_k.dot(S_dependant_k)
                B = 2.0 * S_dependant_k.dot(S_independant_k)
                C =  S_independant_k.dot(S_independant_k) - TrustRegionThreshold**2
                D = B**2 - 4.0 * A * C
            
                try:
                    root_1_tc = (-1.0 * B-math.sqrt(D))/(2*A)
                    root_2_tc = (-1.0 * B+math.sqrt(D))/(2*A)
                except ZeroDivisionError:
                    try:
                        root_1_tc = -1.0 * C / B
                        root_2_tc = root_1_tc
                    except ZeroDivisionError:
                        root_1_tc = sys.maxsize
                        root_2_tc = sys.maxsize
                if root_1_tc > t_key and root_1_tc <= next_t_key:
                    t_threshold = root_1_tc
                else:
                    t_threshold = root_2_tc 


            #Find the first local minimum of the piece-wise quadratic function a * t**2 + b * t
            # this happens in two cases
            # (a) if the quadratic function is convex and its peak is in the interval [t_key, next_t_key]
            # (b) if the quadratic function was decreasing in the last interval and it is increasing now. 
            # check (a)
            if a > 0.0 and -1.0 * b /(2 * a) > t_key and -1.0 * b /(2 * a) < next_t_key:
                t_C_k = -1.0 * b /(2 * a)
                
                if t_C_k > t_threshold:
                    if debug:
                        print "Iter ", i, " Convexity Rechaed the threhsold, T_thre is ",  t_threshold
                    return S_independant_k + S_dependant_k * t_threshold

                else:
                    if debug:
                        print "Convexity"
                    return S_independant_k + S_dependant_k * t_C_k

            # check (b)
            beg_deriavative_sgn = np.sign( 2*a * t_key + b)
            if beg_deriavative_sgn == 0:
                #Check if the quadratic functions peaks coincide with the hitting_times
                if a > 0.0:
                    beg_deriavative_sgn = 1
                elif a == 0.0:
                    beg_deriavative_sgn = 0
                else:
                    beg_deriavative_sgn = -1
                    
            if  end_deriavative_sgn<0 and beg_deriavative_sgn >= 0:
                t_C_k = t_key
                if debug:
                    print "Changin sign"
                return  S_independant_k + S_dependant_k * t_C_k
            end_deriavative_sgn = np.sign( 2*a * next_t_key + b)
            if end_deriavative_sgn == 0:
             #Check if the quadratic functions peaks coincide with the hitting_times
                if a > 0.0:
                    end_deriavative_sgn = -1
                else:
                    end_deriavative_sgn = 1
            if t_threshold < sys.maxsize:
              #If the quadratic function is decreasing in an interval before t_threshold
                if  2*a * t_threshold + b <= 0.0:
                    if debug:
                        print  "Iter ", i, " rechaed threshold, T_thre is ",  t_threshold,TrustRegionThreshold
                    return S_independant_k + S_dependant_k * t_threshold 
                #else:
                #    return S_independant_k      
        return  SparseVector( {} )
             
                                  
                    
                    
            
            
                       

    
        #find x_k + s_k and evaluate the function

        #Do the tests and updates 

        #stop if \|S_k\| is small enough 
          
        
    


class BarrierOptimizer():
    def __init__(self, Pr, logger=None):
        #initialize algorithm parameters 
        initVal = 0.5
        self.eta_s = initVal
        self.omega_s = initVal
        self.alpha_omega = initVal
        self.beta_omega = initVal
        self.beta_eta = 0.25
        self.alpha_lambda = 1.0
        #self.tau = 0.75
        self.tau = 0.95
        self.rho = 0.5
        self.omega_star = 1.e-4
        self.eta_star = 1.e-4
        self.alpha_eta = 1.1 - 1./(1+ self.alpha_lambda)
        self.MU = initVal
        self.OMEGA  = self.omega_s * self.MU**self.alpha_omega 
        self.ETA = self.eta_s * self.MU ** self.alpha_eta
        #Initialize the dual variables LAMBDAS
        constraint_func,constraint_grads, constraint_HESSIAN = Pr.evalFullConstraintsGrad(0 )
        self.LAMBDAS = {}
        #* eps
        eps_margin = 1.e-1
        for constraint in constraint_func:
            if constraint_func[constraint] > 0:
                self.LAMBDAS[constraint] =  (eps_margin / self.MU) ** (1./self.alpha_lambda) 
            else:
                self.LAMBDAS[constraint] = ( (-1.0 * constraint_func[constraint] + eps_margin) / self.MU) ** (1./self.alpha_lambda) 

        #self.LAMBDA_BAR = self.LAMBDAS

        self.innerSolver = boxOptimizer()
        #logger
        self.logger = logger
           


          
         
    def PGD(self, Pr, iterations=100):
    #NOT USED 
        #The follwowing dictionaries keep track of the gradient of the barrier function`s objective 
        self.grad_Psi_VAR  = {}
        #Set the initial point, s.t., the constints functions are maximized 
        #for key in Pr.VAR:
        #     if type(key[1]) == tuple:
                 #Remainder variables 
        #         Pr.VAR[key] = Pr.BOX[key]
        #     else:
                 #Caching variables 
        #         Pr.VAR[key] = 0.0

        #Gradient descent step 
        #w.r.t. constraints
       # constraint_grads, constraint_func = Pr.evalFullConstraintsGrad()
        #w.r.t. objective 
       # obj_grads, obj_func = Pr.evalGradandUtilities()
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
      #      OldObj =  sum( obj_func.values() ) 
      #      for  constraint in constraint_func:
      #          OldObj -=  self.LAMBDAS[constraint] * self.SHIFTS[constraint] * np.log(constraint_func[constraint] + self.SHIFTS[constraint])
      #      self.logger.info("INNER ITERATION %d, current objective value is %f" %(t, OldObj))

            #Optimiality
            non_optimality_VAR = ProjOperator(Pr.VAR, self.grad_Psi_VAR, Pr.BOX)
            non_optimality_norm = squaredNorm( non_optimality_VAR ) 
            self.logger.info("INNER ITERATION %d, current non-optimality is %f." %(t, non_optimality_norm)) 

            print [constraint_func[constraint] + self.SHIFTS[constraint]  for constraint in self.SHIFTS]

            if non_optimality_norm<self.OMEGA:# and np.prod( [constraint_func[constraint] + self.SHIFTS[constraint] >= 0 for constraint in self.SHIFTS] ):
                break
        return constraint_func, non_optimality_norm   
        
    def outerIter(self, Pr, OuterIterations, InnerIterations, debugLevel):
       
        startFromLast = False
        trace = {}
        startTime = time.time()
        for k in range(OuterIterations):
            self.SHIFTS  = dict([(edge, self.MU * (self.LAMBDAS[edge] ** self.alpha_lambda)) for edge in self.LAMBDAS] )
           # if k == OuterIterations -1:
            #print 'shifts are ', self.SHIFTS
            #Set initial point
            self.innerSolver.initialPoint(Pr, self.SHIFTS, startFromLast)
        #    print "Initail point for iter ", k, " is ", Pr.VAR
         #   if k < OuterIterations-1:
        #    print self.OMEGA
            new_LAMBDA_BAR, non_optimality_norm = self.innerSolver.optimizer(Pr, self.LAMBDAS, self.SHIFTS, self.OMEGA, InnerIterations, debug=False, logger=self.logger )
         #   else:
         #       new_LAMBDA_BAR, non_optimality_norm = self.innerSolver.optimizer(Pr, self.LAMBDAS, self.SHIFTS, self.OMEGA, InnerIterations, debug=True, logger=self.logger )
             
            constraint_func, constraint_grads_NULL, constraint_Hessian_NULL = Pr.evalFullConstraintsGrad(0)
            if debugLevel == 'DEBUG':
                print constraint_func
            comp_slack = {}
            comp_slack  = dict([(key, new_LAMBDA_BAR[key] * constraint_func[key]) for key in new_LAMBDA_BAR])

            comp_slack_norm = squaredNorm(comp_slack)
             

            #Evaluate and record stats
            OBJ, CONSTR = Pr.evaluate()
            currentTime = time.time()
            trace[k] = {}
            trace[k]['non_optimilaity'] = non_optimality_norm
            trace[k]['slackness'] = comp_slack_norm
            trace[k]['OBJ'] = OBJ
            trace[k]['CONSTRAINT'] = CONSTR
            trace[k]['time']  = currentTime - startTime
            #log info
            self.logger.info("OUTER ITERATION %d, current non-optimality is %f, current complimentary slackness violation is %f" %(k, non_optimality_norm, comp_slack_norm) )
            self.logger.info("Objective is %f, the ratio of satisfied constraints sum is %f" %(OBJ,  CONSTR) )
            
            if comp_slack_norm < self.eta_star and non_optimality_norm < self.omega_star:
                 break

            #Check condition 
            scaled_comp_slack = {}
            for key in comp_slack:
                try:
                    scaled_comp_slack[key] = comp_slack[key]/(self.LAMBDAS[key]**self.alpha_lambda)
                except ZeroDivisionError:
                    if comp_slack[key] == 0:
                        scaled_comp_slack[key] = 0.0
                    else:
                        raise Exception('Lambda bar is not zero while lambda is zero.')
                        
            if squaredNorm( scaled_comp_slack ) <= self.ETA:
                #Exe. Step 3
                self.logger.info("executing step 3 ")
                self.LAMBDAS = new_LAMBDA_BAR
                self.OMEGA *= (self.MU ** self.beta_omega)
                self.ETA *= (self.MU ** self.beta_eta) 
                startFromLast = True
            else:
                #Exe. Step 4
                self.logger.info("executing step 4, ")
                self.MU *= self.tau
                self.OMEGA = self.omega_s * self.MU ** self.alpha_omega
                self.ETA = self.eta_s * self.MU ** self.alpha_eta
                startFromLast = False 
           
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


    
    problem_instance = Problem.unpickle_cls(args.problem) 
    eps = 1.e-3
       
    #set up logfile
    logger = logging.getLogger('Shifted Barrier Method')
    logger.setLevel(eval("logging."+args.logLevel)) 
    clearFile(args.logfile)
    fh = logging.FileHandler(args.logfile)
    fh.setLevel(eval("logging."+args.logLevel))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
   

    optimizer = BarrierOptimizer(problem_instance, logger)
    trace = optimizer.outerIter(problem_instance, OuterIterations=args.outerIterations, InnerIterations=args.innerIterations, debugLevel=args.debug_level)
    problem_instance.pickle_cls( args.opt_problem )
    
    with open(args.trace_file,'wb') as f:
        pickle.dump((args,trace),f)
    
    
        
        
                


        
        
