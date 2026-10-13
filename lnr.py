class LinearRegression:
    
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr = lr
        self.weights = None 
        self.n_iters = n_iters
        self.bias = None 
    
    def fit(self,X,y):
        