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

class min_max_scaler():
  def __init__(self):
    self.min = None
    self.max = None
  def fit(self,dataset):
    dataset = dataset.to_numpy()
    min = np.min(dataset,axis=0)
    max = np.max(dataset,axis=0)
    self.min = min
    self.max = max
  def transform(self,dataset):
    dataset = dataset.to_numpy()
    return (dataset-self.min)/(self.max-self.min)

class mean_scaler():
  def __init__(self):
    self.min = None
    self.max = None
    self.mean = None
  def fit(self,dataset):
    dataset = dataset.to_numpy()
    min = np.min(dataset,axis=0)
    max = np.max(dataset,axis=0)
    mean = np.mean(dataset,axis=0)
    self.min = min
    self.max = max
    self.mean = mean
  def transform(self,dataset):
    dataset = dataset.to_numpy()
    return (dataset-self.mean)/(self.max-self.min)

class max_abs_scaler():
  def __init__(self):
    self.max = None
  def fit(self,dataset):
    dataset = dataset.to_numpy()
    max = np.max(dataset,axis=0)
    self.max = abs(max)
  def transform(self,dataset):
    dataset = dataset.to_numpy()
    return (dataset/self.max)

class robust_scaler():
  def __init__(self):
    self.median = None
    self.third_quarter = None
    self.first_quarter = None
  def fit(self,dataset):
    dataset = dataset.to_numpy()
    third_quarter = np.percentile(dataset,75,axis=0)
    first_quarter = np.percentile(dataset,25,axis=0)
    median = np.median(dataset,axis=0)
    self.median = median
    self.third_quarter = third_quarter 
    self.first_quarter = first_quarter
  def transform(self,dataset):
    dataset = dataset.to_numpy()
    return (dataset-self.median)/(self.third_quarter-self.first_quarter)