import numpy as np
class LinearRegression():
    def __init__(self):
        self.coeff = None
        self.intercept = None
    def fit(self, x_train,y_train):
        x_train = np.insert(x_train,0,1,axis=1)
        betas = np.dot(np.dot(np.linalg.inv(np.dot(x_train.T,x_train)),x_train.T),y_train)
        self.intercept = betas[0]
        self.coeff = betas[1:]
    def predict(self, x_test):
        return self.intercept + np.dot(self.coeff,x_test.T)
    
class GDRegressor():
    def __init__(self,learn_rate,epochs):
        self.m = None
        self.b = None
        self.learn_rate = learn_rate
        self.epochs = epochs
    def fit(self, x_train,y_train):
        x = x_train.ravel()          # (80,1) -> (80,)
        self.m = 100
        self.b = -120
        for i in range(self.epochs):
            error = y_train - self.m*x - self.b
            slope_b = -2 * np.sum(error)
            slope_m = -2 * np.sum(error * x)
            self.m -= slope_m * self.learn_rate
            self.b -= slope_b * self.learn_rate
        print(self.m, self.b)
            
    def predict(self, x_test):
        x = x_test.ravel()
        return self.b + self.m*x