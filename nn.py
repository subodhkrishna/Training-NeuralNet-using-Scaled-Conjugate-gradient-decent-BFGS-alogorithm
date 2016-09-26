import numpy as np
import time
from scipy import optimize
import matplotlib.pyplot as plt

# X = (smoke tobacco, smoke cannnibis, drink alchol), y = chances of getting lung cancer
X = np.array(([0,0,0], [0,0,1], [0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)
y = np.array(([10], [22], [32], [45], [60], [75], [72], [86]), dtype=float)
X = X/np.amax(X, axis=0)
y = y/100 #Max is 100

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.NetworkInputLayerSize = 3
        self.NetworkOutputLayerSize = 1
        self.NetworkHiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.NetworkInputLayerSize,self.NetworkHiddenLayerSize)
        self.W2 = np.random.randn(self.NetworkHiddenLayerSize,self.NetworkOutputLayerSize)
        
    def forwardNetwork(self, X):
        #one run
        self.z2 = np.dot(X, self.W1)        #compute activation values for neurons in hidden layer
        self.a2 = self.sigmoid(self.z2)     #apply sigmoid function to hiddenlayer neuron
        self.z3 = np.dot(self.a2, self.W2)  #compute activation value for output neuron
        yprime = self.sigmoid(self.z3)      #apply sigmoid for the output
        return yprime
        
    def sigmoid(self, z):
        #Threshold function
        return 1/(1+np.exp(-z))
    
    def sigmoidGradient(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def performanceFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yprime = self.forwardNetwork(X)
        J = 0.5*sum((y-self.yprime)**2)
        return J
        
    def performanceFunctionPrime(self, X, y):
        #Compute derivative with respect to W1 and W2 for a given X and y:
        self.yprime = self.forwardNetwork(X)
        d2 = np.multiply(-(y-self.yprime), self.sigmoidGradient(self.z3))
        dPdW2 = np.dot(self.a2.T, d2)
        d1 = np.dot(d2, self.W2.T)*self.sigmoidGradient(self.z2)
        dPdW1 = np.dot(X.T, d1)  
        return dPdW1, dPdW2

     #Helper Functions for interacting with other classes fetching and setting the wights for training:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.NetworkHiddenLayerSize * self.NetworkInputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.NetworkInputLayerSize , self.NetworkHiddenLayerSize))
        W2_end = W1_end + self.NetworkHiddenLayerSize*self.NetworkOutputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.NetworkHiddenLayerSize, self.NetworkOutputLayerSize))
        
    def findGradients(self, X, y):
        dPdW1, dPdW2 = self.performanceFunctionPrime(X, y)
        return np.concatenate((dPdW1.ravel(), dPdW2.ravel()))

class back_prop(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.performanceFunction(self.X, self.y))   
        
    def functionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.performanceFunction(X, y)
        grad = self.N.findGradients(X,y)        
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y
        #Make empty list to store costs:
        self.J = []
        params0 = self.N.getParams()
        options = {'maxiter': 200, 'disp' : True}
        # using the SciPy function to calculate approximate solution for the system of equations
        _res = optimize.minimize(self.functionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)
        self.N.setParams(_res.x)
        #print the final solution fo the weights        
        print (_res.x)
        self.optimizationResults = _res

    def selftrain(self,X,y):
        learningrate = 0.7 #scaling value for the gradient to avoid local minimum
        count = 1
        threshold = 0.0001 #cost we want to achice
        self.X = X
        self.y = y
        #Make empty list to store costs:
        self.J = []
        params0 = self.N.getParams()
        self.J.append(self.N.performanceFunction(X,y))
        #initialize with the random weights and forwardNetwork the network
        while True:
            dPdW1, dPdW2 = self.N.performanceFunctionPrime(X,y)
            self.N.W1 = self.N.W1 - learningrate*dPdW1
            self.N.W2 = self.N.W2 - learningrate*dPdW2
            self.N.forwardNetwork(X)
            self.J.append(self.N.performanceFunction(X,y))
            #print(self.J[count])
            if (abs(self.J[count]) < 0.0001):
                break
            count = count +1
        params = self.N.getParams()
        print (params)
        print ("\tFinal cost : %f"%self.J[count-1])
        print ("\tIterations : %d"%(count-1))


msec = lambda: int(round(time.time() * 1000))

#Instance 1 for training using the BFGS algorithm   
NN1 = Neural_Network()
T1 = back_prop(NN1)     
starttime1  = msec()    #start timer
T1.train(X,y)           #train the network 
endtime1 = msec()       #end timer
tottime1 = endtime1 - starttime1
#print("Final COst %f\n"%T1.optimizationResults.x)
print ( "BDFS time %d milli seconds\n"%tottime1)

#Instance 1 for training using the my algorithm
NN2 = Neural_Network()
T2 = back_prop(NN2)
starttime  = msec() #start timer
T2.selftrain(X,y)   #train the network 
endtime = msec()    #end timer
tottime = endtime - starttime
print ( "my method time %d milli seconds\n"%tottime)

x1 = range(len(T1.J))
x2 = range(len(T2.J))

plt.plot(x2,T2.J,'r.-', label="MY ALGORITHM")
plt.title('COST FUNCTION GRADIENT DECENT')
plt.ylabel("Cost")
plt.legend(loc='upper left')
plt.xlabel("Iterations")
plt.show()

plt.plot(x1,T1.J,'ko-', label="BFGS")
plt.title('COST FUNCTION GRADIENT DECENT')
plt.ylabel("Cost")
plt.legend(loc='upper left')
plt.xlabel("Iterations")
plt.show()