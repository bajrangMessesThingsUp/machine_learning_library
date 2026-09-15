from scaling import *
from encoding import *
import numpy as np
import pandas as pd

class ColumnTransformer():
  def __init__(self,transformer):
    self.transformer = transformer
    self.trn_dict = dict()
    self.new_arr = None
    for i in self.transformer:
      self.trn_dict[i[0]] = i[1]
  def fit(self, data):
    for i in self.transformer:
      self.trn_dict[i[0]]=self.trn_dict[i[0]].fit(data[i[2]])
  def transform(self,data):
    c=0
    for i in self.transformer:
      if(c==0):
        self.new_arr = np.array(self.trn_dict[i[0]].transform(data[i[2]]))
        c=-1
      else:
        self.new_arr=np.column_stack((self.new_arr,self.trn_dict[i[0]].transform(data[i[2]])))
        print(self.new_arr.shape)
    return self.new_arr