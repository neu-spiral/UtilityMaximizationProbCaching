class BarrierOptimizer():
    def __init__(self):
        #initialize algorithm parameters 
        initval = 0.5
        self.eta_s = initVal
        slef.omega_s = initVal
        self.alpha_omega = initVal
        self.beta_omega = initVal
        self.beta_eta = initVal
        self.alpha_lambda = initVal
        self.tau = 0.5
        self.rho = 0.5
        self.omega_star = 1.e-4
        self.eta_star = 1.e-4
        self.alpha_eta = 1.1 - 1./(1+ self.alpha_lambda)
        self.MU = initval
        self.OMEGA  = slef.omega_s * self.MU**self.alpha_omega 
        self.ETA = self.eta_s * self.MU ** self.alpha_eta

    def initLambda(self, Pr):
       grads, edge_func = Pr.evalGradandConstraints()
       self.LAMBDAS = {}
       for edge in edge_func:
           if edge_func[edge] > 0:
                self.LAMBDAS[edge] = 0.0
           else:
                self.LAMBDAS[edge] = (-1.0 * edge_func[edge] / self.MU) ** (1./self.alpha_lambda)
           
        caching_grads, caching_func = Pr.evalGradandCapcityConstraints()
        #To be continued


    def PGD(self, Pr, iterations=100):
        for t in range(iterations):
            step_size = 1./(t+2)
            #Gradient descent setp


            #w.r.t. constraints 
            constraint_grads, constraint_func = Pr.evalGradandConstraints()
            for edge in constraint_grads:
                LAMBDA_BAR =  self.LAMBDAS[edge] * self.SHIFTS[edge] / (constraint_func[edge] + self.SHIFTS[edge])
                for index in grads[edge]:
                    grad_index = -1.0 * LAMBDA_BAR * constraint_grads[edge][index] 
                    if type(index) == list:
                        Pr.REM[index] -= step_size * grad_index
                    else:
                        Pr.VAR[index] -= step_size * grad_index
            #w.r.t. caching constraints 
            caching_grads, caching_func = Pr.evalGradandCapcityConstraints()
            for node in caching_grads:
                LAMBDA_BAR
            


             #w.r.t. objective 
             obj_grads, obj_func = Pr.evalGradandUtilities() 
             for index in obj_grads:
                  Pr.REM[index] -= step_size * obj_grads[index]

            

              #Projections 
              
                         
                    
                 
            
     
        
    def  outerIter(self, Pr):
        #Initialize the dual variables
        self.initLambda(Pr)

        #Initialize the shifts
        self.SHIFTS  = dict([(edge, self.MU * (self.LAMBDAS[edge] ** self.alpha_lambda)) for edge in self.LAMBDAS])
        
        
                


        
        
