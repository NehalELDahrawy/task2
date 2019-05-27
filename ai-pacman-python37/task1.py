import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)
 
#%config InlineBackend.figure_formats = {'pdf',}
 #%matplotlib inline  

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')


def warmUpExercise():
    return(np.identity(5))
    
    
    

   #data = np.loadtxt('data/ex1data1.txt', delimiter=',')

  # X = np.c_[np.ones(data.shape[0]),data[:,0]]
   
   #y = np.c_[data[:,1]]


  #plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
  #plt.xlim(4,24)
  #plt.xlabel('Population of City in 10,000s')
  #plt.ylabel('Profit in $10,000s');   
  
  #def computeCost(X, y, theta=[[0],[0]]):
   # m = y.size # n
    #J = 0 #rss
    
    #h = X.dot(theta) #f or y_hat
    
    #J = 1/(2*m)*np.sum(np.square(h-y)) #rss
    
    #return(J)
    
    