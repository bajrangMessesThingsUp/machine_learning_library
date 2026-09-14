import pandas as pd
import numpy as np

class OridnalEncoder():
  def __init__(self,categories: list[list]):
    self.categories = categories
  def fit(self,dataset):
    for i in self.categories:
      for j in i:
        dataset[dataset==j] = i.index(j)
    return dataset 