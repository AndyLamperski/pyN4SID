import numpy as np
import scipy.linalg as la

try:
    import cvxpy as cvx
    hasCvx = True

except ImportError:
    hasCvx = False
    


def generalizedPlant(A,B,C,D,Cov,dt):
    CovChol = la.cholesky(Cov,lower=True)
    NumStates = len(A)

    B1 = CovChol[:NumStates,:]
    B2 = B

    Bbig = np.hstack((B1,B2))

    D1 = CovChol[NumStates:,:]
    D2 = D
    Dbig = np.hstack((D1,D2))

    P = (A,Bbig,C,Dbig,dt)
    return P

def block2mat(Mblock):
    Nr,Nc,bh,bw = Mblock.shape
    M = np.zeros((Nr*bh,Nc*bw))
    for k in range(Nr):
        M[k*bh:(k+1)*bh] = np.hstack(Mblock[k])
        
    return M

def blockTranspose(M,blockHeight,blockWidth):
    """
    Switches block indices without transposing the blocks
    """
    r,c = M.shape
    Nr = r / blockHeight
    Nc = c / blockWidth
    Mblock = np.zeros((Nr,Nc,blockHeight,blockWidth))
    for i in range(Nr):
        for j in range(Nc):
            Mblock[i,j] = M[i*blockHeight:(i+1)*blockHeight,j*blockWidth:(j+1)*blockWidth]
            
            
    MtBlock = np.zeros((Nc,Nr,blockHeight,blockWidth))
    for i in range(Nr):
        for j in range(Nc):
            MtBlock[j,i] = Mblock[i,j]
            
    return block2mat(MtBlock)

def blockHankel(Hleft,Hbot=None,blockHeight=1):
    """
    Compute a block hankel matrix from the left block matrix and the optional bottom block matrix
    
    Hleft is a matrix of dimensions (NumBlockRows*blockHeight) x blockWidth
    
    Hbot is a matrix of dimensions blockHeight x (NumBlockColumns*blockWidth)
    """
    
    blockWidth = Hleft.shape[1]
    if Hbot is None:
        Nr = len(Hleft) / blockHeight
        Nc = Nr
    else:
        blockHeight = len(Hbot)
        Nr = len(Hleft) / blockHeight
        Nc = Hbot.shape[1] / blockWidth
        
    LeftBlock = np.zeros((Nr,blockHeight,blockWidth))
    
    for k in range(Nr):
        LeftBlock[k] = Hleft[k*blockHeight:(k+1)*blockHeight]
        
    
        
    # Compute hankel matrix in block form
    MBlock = np.zeros((Nr,Nc,blockHeight,blockWidth))
    
    for k in range(np.min([Nc,Nr])):
        # If there is a bottom block, could have Nc > Nr or Nr > Nc
        MBlock[:Nr-k,k] = LeftBlock[k:]
        
        
    if Hbot is not None:
        BotBlock = np.zeros((Nc,blockHeight,blockWidth))
        for k in range(Nc):
            BotBlock[k] = Hbot[:,k*blockWidth:(k+1)*blockWidth]
            
        for k in range(np.max([1,Nc-Nr]),Nc):
            MBlock[Nr-Nc+k,Nc-k:] = BotBlock[1:k+1]
            
    
    # Convert to a standard matrix
    M = block2mat(MBlock)
        
    return M

def getHankelMatrices(x,NumRows,NumCols,blockWidth=1):
    # For consistency with conventions in Van Overschee and De Moor 1996, 
    # it is assumed that the signal at each time instant is a column vector
    # and the number of samples is the number of columns.
    
    bh = len(x)
    bw = 1
    xPastLeft = blockTranspose(x[:,:NumRows],blockHeight=bh,blockWidth=bw)
    XPast = blockHankel(xPastLeft,x[:,NumRows-1:NumRows-1+NumCols])
    
    xFutureLeft = blockTranspose(x[:,NumRows:2*NumRows],blockHeight=bh,blockWidth=bw)
    XFuture = blockHankel(xFutureLeft,x[:,2*NumRows-1:2*NumRows-1+NumCols])
    return XPast,XFuture

def preProcess(u,y,NumDict):
    
    NumInputs = u.shape[0]
    NumOutputs = y.shape[0]
    NumRows = NumDict['Rows']
    NumCols = NumDict['Columns']
    NSig = NumDict['Dimension']
    UPast,UFuture = getHankelMatrices(u,NumRows,NumCols)
    YPast,YFuture = getHankelMatrices(y,NumRows,NumCols)
    Data = np.vstack((UPast,UFuture,YPast))
    L = la.lstsq(Data.T,YFuture.T)[0].T
    Z = np.dot(L,Data)
    DataShift = np.vstack((UPast,UFuture[NumInputs:],YPast))
    LShift = la.lstsq(DataShift.T,YFuture[NumOutputs:].T)[0].T
    ZShift = np.dot(LShift,DataShift)

    L1 = L[:,:NumInputs*NumRows]
    L3 = L[:,2*NumInputs*NumRows:]

    LPast = np.hstack((L1,L3))
    DataPast = np.vstack((UPast,YPast))

    U, S, Vt = la.svd(np.dot(LPast,DataPast))
    
    Sig = np.diag(S[:NSig])
    SigRt = np.diag(np.sqrt(S[:NSig]))
    Gamma = np.dot(U[:,:NSig],SigRt)
    GammaLess = Gamma[:-NumOutputs]

    GammaPinv = la.pinv(Gamma)
    GammaLessPinv = la.pinv(GammaLess)

    GamShiftSolve = la.lstsq(GammaLess,ZShift)[0]


    GamSolve = la.lstsq(Gamma,Z)[0]
    GamData = np.vstack((GamSolve,UFuture))

    GamYData = np.vstack((GamShiftSolve,YFuture[:NumOutputs]))
    # Should probably move to a better output structure
    # One that doesn't depent so heavily on ordering

    GammaDict = {'Data':GamData,
                 'DataLess':GammaLess,
                 'DataY':GamYData,
                 'Pinv': GammaPinv,
                 'LessPinv': GammaLessPinv}
    return GammaDict,S

def postProcess(K,GammaDict,NumDict):
    GamData = GammaDict['Data']
    GamYData = GammaDict['DataY']

    rho = GamYData - np.dot(K,GamData)

    NSig = NumDict['Dimension']
    AID = K[:NSig,:NSig]
    CID = K[NSig:,:NSig]
    

    NumCols = NumDict['Columns']
    CovID = np.dot(rho,rho.T) / NumCols
    
    # Now we must construct B and D

    GammaPinv = GammaDict['Pinv']
    GammaLessPinv = GammaDict['LessPinv']
    AC = np.vstack((AID,CID))
    L = np.dot(AC,GammaPinv)

    NumRows = NumDict['Rows']
    NumOutputs = NumDict['Outputs']
    M = np.zeros((NSig,NumRows*NumOutputs))
    M[:,NumOutputs:] = GammaLessPinv
    Mleft = blockTranspose(M,NSig,NumOutputs)
    LtopLeft = blockTranspose(L[:NSig],NSig,NumOutputs)
    NTop = blockHankel(Mleft,blockHeight=NSig) - blockHankel(LtopLeft,blockHeight=NSig)

    LbotLeft = blockTranspose(L[NSig:],NumOutputs,NumOutputs)
    NBot= -blockHankel(LbotLeft,blockHeight=NumOutputs)
    NBot[:NumOutputs,:NumOutputs] = NBot[:NumOutputs,:NumOutputs] + np.eye(NumOutputs)

    GammaLess = GammaDict['DataLess']
    N = np.dot(np.vstack((NTop,NBot)),la.block_diag(np.eye(NumOutputs),GammaLess))

    NumInputs = NumDict['Inputs']
    KsTop = np.zeros((NSig*NumRows,NumInputs))
    KsBot = np.zeros((NumOutputs*NumRows,NumInputs))

    Kr = K[:,NSig:]
    for k in range(NumRows):
        KsTop[k*NSig:(k+1)*NSig] = Kr[:NSig,k*NumInputs:(k+1)*NumInputs]
        KsBot[k*NumOutputs:(k+1)*NumOutputs] = Kr[NSig:,k*NumInputs:(k+1)*NumInputs]
    
    Ks = np.vstack((KsTop,KsBot))

    DB = la.lstsq(N,Ks)[0]

    BID = DB[NumOutputs:]
    DID = DB[:NumOutputs]

    return AID,BID,CID,DID,CovID


def N4SID(u,y,NumRows,NumCols,NSig,require_stable=False):
    """
    A,B,C,D,Cov,Sigma = N4SID(u,y,NumRows,NumCols,n,require_stable=False)

    Let NumVals be the number of input and output values available
    In this case:

    u - NumInputs x NumVals array of inputs
    y - NumOutputs x NumVals array of outputs

    NumRows - Number of block rows in the past and future block Hankel matrices
    NumCols - Number of columns in the past and future block Hankel matrices

    n - desired state dimension.

    For the algorithm to work, you must have:

    NumVals >= 2*NumRows + NumCols - 1

    Returns

    A,B,C,D - the state space realization from inputs to outputs
    
    Cov - the joint covariance of the process and measurement noise

    Sigma - the singular values of the oblique projection of 
            row space of future outputs along row space of 
            future inputs on the row space of past inputs and outputs.
            
            Examining Sigma can be used to determine the required state 
            dimension

    require_stable - An optional boolean parameter. Default is False
                     If False, the standard N4SID algorithm is used
                     If True, the state matrix, A, 
                     will have spectral radius < 1.
                     In order to run with require_stable=True, cvxpy 
                     must be installed. 
           

    """
    NumInputs = u.shape[0]
    NumOutputs = y.shape[0]

    
    NumDict = {'Inputs': NumInputs,
               'Outputs': NumOutputs,
               'Dimension':NSig,
               'Rows':NumRows,
               'Columns':NumCols}
    
    GammaDict,S = preProcess(u,y,NumDict)


    GamData = GammaDict['Data']
    GamYData = GammaDict['DataY']

    if not require_stable:
        K = la.lstsq(GamData.T,GamYData.T)[0].T
    else:

        Kvar = cvx.Variable(NSig+NumOutputs,NSig+NumInputs*NumRows)

        Avar = Kvar[:NSig,:NSig]

        Pvar = cvx.Semidef(NSig)

        LyapCheck = cvx.vstack(cvx.hstack(Pvar,Avar),
                               cvx.hstack(Avar.T,np.eye(NSig)))

        Constraints = [LyapCheck>>0,Pvar << np.eye(NSig)]

        diffVar = GamYData - Kvar*GamData
        objFun = cvx.norm(diffVar,'fro')
        Objective = cvx.Minimize(objFun)

        Prob = cvx.Problem(Objective,Constraints)

        result = Prob.solve()

        K = Kvar.value

        
    
    AID,BID,CID,DID,CovID = postProcess(K,GammaDict,NumDict)

    
    
    return AID,BID,CID,DID,CovID,S
