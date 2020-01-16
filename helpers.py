import numpy as np
from scipy.stats import rv_discrete
def Project2Box(X, U):
    '''
        Project elemnts of the dict X on the box defined by the lower bound L 
        and the upper bound U (both dict)
    '''

    L = dict( [(key, 0.0) for key in X] )
    for key in X:
        if X[key] < L[key]:
            X[key] = L[key]
        elif X[key] > U[key]:
            X[key] = U[key]
def Proj2PositiveOrthant(X):
    for key in X:
       if X[key] < 0:
           X[key] = 0.0        
def ProjOperator(X, V, U):
    'The projector operator defined in Eq. (2,9)'
    XminusV = dict( [(key, X[key] - V[key]) for key in X] )
    Project2Box(XminusV, U)
    
    P_dict = {}
    for key in XminusV:
        P_dict[key] = X[key] - XminusV[key] 
    return P_dict
def clearFile(file):
    "Delete all contents of a file"	
    with open(file,'w') as f:
   	f.write("")

def squaredNorm(X):
    'Compute the ell_2 norm for a dict'
    return np.sqrt( sum([val**2 for val in X.values()]) )
       
def reverseDict(d):
    "Reteun an inverse dictionary of the given dict d"
    if len(d.values()) != len(set(d.values())):
        raise ValueError('Dictionary is invertible.')
    return dict( [(d[key], key) for key in d] )


def constructDistribution(d,cap):
    epsilon = 1.e-3

    #Remove very small values, rescale the rest
    dd = dict( (key,d[key]) for key in d if d[key]>epsilon  )
    keys, vals = zip( *[ (key,d[key]) for key in dd] )
    ss = sum(vals)
    vals = [ val/ss*cap for val in vals]
    dd = dict(zip(keys,vals))

    intvals = [int(np.round(x/epsilon)) for x in vals]
    intdist = int(1/epsilon)
    intdd = dict(zip( keys,intvals  ))

    s = {}
    t = {}
    taus = []
    sumsofar = 0
    for item in keys:
        s[item] = sumsofar
        t[item] = sumsofar + intdd[item]
        taus.append(t[item] % intdist )
        sumsofar = t[item]

    #print s,t,taus
    taus = sorted(set(taus))
    #print taus

    if intdist not in taus:
        taus.append(intdist)

    placements = {}
    prob = {}

    for i in range(len(taus)-1):
        x = []
        t_low = taus[i]
        t_up = taus[i+1]

        diff = t_up -t_low

        for ell in range(int(cap)):
           lower = ell*intdist + t_low
           upper = ell*intdist + t_up
           for item in keys:
                #print lower,upper,' inside ', s[item],t[item], '?',
                if lower>=s[item] and upper<=t[item]:
                    x.append(item)
                #    print ' yes'
                #else: print ' no'
        prob[i] = 1.*diff/intdist
        placements[i] = x

    totsum = np.sum(prob.values())
    if not np.allclose(totsum,1):
        for i in prob:
            prob[i] = 1.*prob[i]/totsum
        # round to 1


    #    print "Placements ",placements,"with prob",prob,"summing to",np.sum(prob.values())


    return placements,prob, rv_discrete(values=(prob.keys(),prob.values()))
