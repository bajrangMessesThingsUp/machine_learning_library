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

class onehotencoder():
    def __init__(self):
        self.categories = []
    def fit(self, dataset):
      for i in range(len(dataset.columns)):
        self.categories.append(dataset[dataset.columns[i]].unique())
    def transform(self,dataset):
      cat_list=[]
      for i in range(len(self.categories)):
        for j in self.categories[i]:
          cat_list.append(j)
      new_df = pd.DataFrame(columns=cat_list)
      for i in range(len(self.categories)):
        for j in self.categories[i]:
          new_df[j] = (dataset.iloc[:,i]==j).astype(int)
      return new_df