import numpy as np
import numpy.matlib as mlib

def affinityPropagationR(data, similarityFunction, lam = 0.5, maxiter = 100):
    ''' 
    data: database (n x m matrix)
    lam: dumping factor
    maxiter: maximum number of iterations
    '''
    
    realmax = np.inf
    n = data.shape[0] # database size
    
    S = similarityFunction(data) # similarity matrix
    R = np.zeros([n,n]) # responsibility matrix
    A = np.zeros([n,n]) # availability matrix 
    
    # Compute similarities #IMPLEMENT THIS!!!
    
    
    ddd
    for it in range(1,maxiter):
        #Compute responsibilities
        Rold = R
        AS = A+S
        
        y = np.max(AS,0)
        idx = np.argmax(AS, 0) 
        
        for i in range(1,n):
            AS[i,idx[i]] = -realmax #check this
        
        y2 = np.max(AS,0)
        #idx2 = np.argmax(AS, 0) # Dead code (I think) delete after tests 
        
        # Compute responsibilities
        R = S - np.transpose(mlib.repmat(y,n,1)) 
        
        for i in range(1,n):
            R[i,idx[i]]  = S[i,idx[i]] - y2[i]
            
        # Dampen responsibilities
        R = (1-lam) * R+lam*Rold;
        
        # Compute availabilities 
        Aold = A
        
        # Get the matrix with positive responsibility values 
        Rp = np.maximum(R,np.zeros(n,n))
        
        # Copy the main diagonal  
        for i in range(1,n):
            Rp[i,i] = R[i,i]
            
        A = mlib.repmat(np.sum(Rp, 0),n,1) - Rp

        # Get diagonal of A
        dA = np.diag(A)
        A = np.maximum(A,np.zeros(n,n))
        
        for i in range(1,n):
            A[i,i] = dA[i]
        
        # Dampen availabilities
        A = (1-lam)*A+lam*Aold
         
    E = R+A #Pseudo marginals
    idx = np.argwhere(np.diag(E)>0) #Indices of exemplars
    idx = idx[:,0]
    c = np.argmax(S[:,idx],0)
    exemplars = idx
           
    return (c,exemplars)