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
        self.coef = None
        self.intercept = None
        self.learn_rate = learn_rate
        self.epochs = epochs
    def fit(self, x_train,y_train):
        self.intercept = 0
        self.coef = np.ones(x_train.shape[1])
        slope_coef = np.ones(x_train.shape[1])
        for i in range(self.epochs):
            y_i = self.intercept + np.dot(x_train, self.coef)
            error = y_train - y_i
            slope_coef = ((-2/x_train.shape[0])*np.dot(error.T,x_train))
            slope_intercept = ((-2) * np.mean(error))
            self.coef -= slope_coef * self.learn_rate
            self.intercept -= slope_intercept * self.learn_rate
        print(self.coef)
        print(self.intercept)
        
            
    def predict(self, x_test):
        y_pred = np.dot(x_test,self.coef)+self.intercept
        return y_pred
    
class SGDRegressor():
    def __init__(self,learn_rate,epochs):
        self.coef = None
        self.intercept = None
        self.learn_rate = learn_rate
        self.epochs = epochs
    def fit(self, x_train,y_train):
        self.intercept = 0
        self.coef = np.ones(x_train.shape[1])
        slope_coef = np.ones(x_train.shape[1])
        for i in range(self.epochs):
            for row in range(x_train.shape[0]):
                j = np.random.randint(0,x_train.shape[0])
                y_i = self.intercept + np.dot(x_train[j], self.coef)
                error = y_train[j] - y_i
                slope_coef = ((-2)*np.dot(error.T,x_train[j]))
                slope_intercept = ((-2) * error)
                self.coef -= slope_coef * self.learn_rate
                self.intercept -= slope_intercept * self.learn_rate
        print(self.coef)
        print(self.intercept)
        
            
    def predict(self, x_test):
        y_pred = np.dot(x_test,self.coef)+self.intercept
        return y_pred