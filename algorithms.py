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