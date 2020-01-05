import numpy as np
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

