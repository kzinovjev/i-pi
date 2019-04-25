"""Compressed sparse row and compressed sparse column sparse matrix implementation"""

import numpy as np

##########################################################################################

#Load sparse matrix file in binary format
def load_npz(file):
  l = np.load(file)
  if l['kind'] == 'csr':
    return csr_matrix(a=l['a'], ia=l['ia'], ja=l['ja'], m=l['m'], n=l['n']) 
  else:
    return csc_matrix(a=l['a'], ia=l['ia'], ja=l['ja'], m=l['m'], n=l['n']) 

##########################################################################################

#Save sparse matrix file in binary format
def save_npz(file, matrix, compressed=True):
  a = matrix.a
  ia = matrix.ia
  ja = matrix.ja
  m = matrix.m
  n = matrix.n
  kind = matrix.kind
  if compressed == True:
    np.savez_compressed(file, a=a, ia=ia, ja=ja, m=m, n=n, kind=kind)
  else: 
    np.savez(file, a=a, ia=ia, ja=ja, m=m, n=n, kind=kind)

##########################################################################################

#Generic base class
class sparse_matrix(object):

  def __add__(self, b):
    densea = self.toarray()
    denseb = b.toarray()
    return csr_matrix(densea + denseb)

  def __sub__(self, b):
    densea = self.toarray()
    denseb = b.toarray()
    return csr_matrix(densea - denseb)

  def __mul__(self, b):
    densea = self.toarray()
    denseb = b.toarray()
    return csr_matrix(densea*denseb)

  #dot product
  def dot(self, b):
    densea = self.toarray()
    denseb = b.toarray()
    return csr_matrix(np.dot(densea, denseb))

  #fraction of non-zero elements in matrix
  def density(self):
    return float(len(self.a))/float(self.m*self.n)

##########################################################################################

#Compressed sparse row format
class csr_matrix(sparse_matrix):

  def __init__(self, nparray=None, **kwargs):
    self.kind = 'csr'
    #Construct by converting a numpy array to csr
    if 'a' not in kwargs:
      self.m = nparray.shape[0]
      self.n = nparray.shape[1]
      nonzeroids = np.where(nparray != 0)
      self.a = nparray[nonzeroids]
      self.ia = np.zeros(self.m+1, dtype=np.int64)
      for row in nonzeroids[0]:
        self.ia[row+1] += 1
      self.ia = np.cumsum(self.ia)
      self.ja = nonzeroids[1]
    #Construct from sparse format arrays
    else:
      self.a = kwargs['a']
      self.ia = kwargs['ia']
      self.ja = kwargs['ja']
      self.m = kwargs['m']
      self.n = kwargs['n']

  #Convert to dense format
  def toarray(self):
    nparray = np.zeros((self.m, self.n), dtype=self.a.dtype)
    for row in xrange(self.m):
      nparray[row, self.ja[self.ia[row]:self.ia[row+1]]] = \
        self.a[self.ia[row]:self.ia[row+1]]
    return nparray

##########################################################################################

#Compressed sparse column format
class csc_matrix(sparse_matrix):

  def __init__(self, nparray=None, **kwargs):
    self.kind = 'csc'
    #Construct by converting a numpy array to csr
    if 'a' not in kwargs:
      self.m = nparray.shape[0]
      self.n = nparray.shape[1]
      nonzeroids = np.where(nparray.T != 0)
      self.a = nparray.T[nonzeroids]
      self.ia = np.zeros(self.n+1, dtype=np.int64)
      for col in nonzeroids[0]:
        self.ia[col+1] += 1
      self.ia = np.cumsum(self.ia)
      self.ja = nonzeroids[1]
    #Construct from sparse format arrays
    else:
      self.a = kwargs['a']
      self.ia = kwargs['ia']
      self.ja = kwargs['ja']
      self.m = kwargs['m']
      self.n = kwargs['n']

  #Convert to dense format
  def toarray(self):
    nparray = np.zeros((self.m, self.n), dtype=self.a.dtype)
    for col in xrange(self.n):
      nparray[self.ja[self.ia[col]:self.ia[col+1]], col] = \
        self.a[self.ia[col]:self.ia[col+1]]
    return nparray

##########################################################################################