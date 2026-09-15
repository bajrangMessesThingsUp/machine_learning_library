import pandas as pd
import numpy as np

class OridnalEncoder():
  def __init__(self,categories: list[list]):
    self.categories = categories
  def fit(self,dataset):
    for i in range(dataset.shape[1]):
        if(sum(~dataset.iloc[:,i].isin(self.categories[i]))!=0):
            raise ValueError
  def transform(self,dataset):
    for i in self.categories:
      for j in i:
        dataset[dataset==j] = i.index(j)
    return dataset.to_numpy() 

class LabelEncoder():
    def __init__(self):
        self.categories = []
    
    def fit(self, dataset):
        for i in dataset.unique():
            self.categories.append(i)
    def transform(self,dataset):
        for i in self.categories:
                dataset[dataset==i] = self.categories.index(i)
        return dataset.to_numpy() 
