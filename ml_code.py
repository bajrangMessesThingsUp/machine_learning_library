import pandas as pd
import numpy as np

class standard_scaler():
  def __init__(self):
    self.mean = None
    self.std_dev = None
  def fit(self,dataset):
    dataset = dataset.to_numpy()
    mean = np.mean(dataset,axis=0)
    std_dev = np.std(dataset,axis=0)
    self.mean = mean
    self.std_dev = std_dev
  def transform(self,dataset):
    dataset = dataset.to_numpy()
    return (dataset-self.mean)/self.std_dev