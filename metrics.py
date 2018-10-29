""" An implementation of some of Vargas' measures of recommendation diversity.

For more information:
	@phdthesis{vargas2015novelty,
	  title={Novelty and diversity evaluation and enhancement in recommender systems},
	  author={Vargas, S},
	  year={2015},
	  school={Ph. D. thesis}
	}

Todo:
	* Proper commenting
	* Clean up

"""

from __future__ import division
import numpy as np
from scipy import spatial
from scipy import stats
from scipy.stats import norm
from sklearn import metrics
import random
import time

def EPC(Rec,RecAsMatrix,M,U_,Rtest):
	""" Expected Popularity Complement (EPC). 

	"""

	A = []
	for u in range(U_):
		if u not in Rec.keys(): continue
		Cu = 1/np.sum([disc(i) for i,item in enumerate(Rec[u])])
		sum_ = 0
		for i,item in enumerate(Rec[u]):
			sum_+= (1 - np.sum(Ui(item,M))/U_)*disc(i)*Prel(item,u,Rtest)
		A.append(sum_*Cu)
	#print("EPC:",np.mean(A))
	return (np.mean(A),np.std(A))

def EFD(Rec,RecAsMatrix,M,U_,Rtest):
	""" Expected Free Discovery (EFD).

	"""

	A = []
	for u in range(U_):
		if u not in Rec.keys(): continue
		Cu = 1/np.sum([disc(i) for i,item in enumerate(Rec[u])])
		sum_ = 0
		for i,item in enumerate(Rec[u]):
			top = np.sum(Ui(item,M))
			bottom = np.sum(np.sum(M))
			sum_+= np.log2(top/bottom)*disc(i)*Prel(item,u,Rtest)
		A.append(sum_*(-Cu))
	#print("EFD:",np.mean(A))
	return np.mean(A),np.std(A)
			

def EPD(Rec,RecAsMatrix,M,U_,Rtest,dist):
	""" Expected Profile Distance (EPD).

	"""
	A = [] 
	for u in range(U_):
		if u not in Rec.keys(): continue
		Cu = 1/np.sum([disc(i) for i,item in enumerate(Rec[u])])
		Cu_ = np.sum([Prel(i,u,Rtest) for i in np.where(Iu(u, M)>=1)[0]])
		Iuu  = np.where(Iu(u, M)>=1)[0]
		sum_ = 0
		for i,item in enumerate(Rec[u]):
			for itemj in Iuu:
				sum_ += dist[item,itemj]*disc(i)*Prel(item,u,Rtest)*Prel(itemj,u,Rtest)
		A.append((Cu/Cu_)*sum_)
	return np.mean(A),np.std(A)

def EILD(Rec,RecAsMatrix,M,U_,Rtest,dist):
	""" Expected Intra-List Distance (EILD)

	Not sure if this works correctly. Not used.

	"""

	A = [] 
	for u in range(U_):
		if u not in Rec.keys(): continue
		Cu = 1/np.sum([disc(i) for i,item in enumerate(Rec[u])])
		sum_ = 0
		for i,item in enumerate(Rec[u]):
			Ci = Cu/np.sum([disc(max(0,j-i))*Prel(itemj,u,Rtest) for j,itemj in enumerate(Rec[u])])
			for j,itemj in enumerate(Rec[u]):
				if j>i:
					sum_ += dist[item,itemj]*disc(i)*disc(max(0,j-i))*Prel(item,u,Rtest)*Prel(itemj,u,Rtest)*Ci
		A.append(sum_)
	return np.mean(A), np.std(A)


def ILD(Rec,RecAsMatrix,M,U_,dist):
	""" Intra-List distance (ILD)

	"""

	allR = np.where(np.sum(RecAsMatrix,axis=0)>=1)[0]
	sum_ = 0
	for item in allR:
		for itemj in allR:
			sum_ += dist[item,itemj]
	R_ = np.sum(np.sum(RecAsMatrix))
	return (1/(R_*(R_-1)))*sum_, 0


# User and item interaction profiles
def Iu(u, M):
	return M[u,:]

def Ui(i, M):
	return M[:,i]

def Prel(i, u, Mr):
	if Mr[u,i]>=1: return 1
	else: return 0.01

# User rec profile
def R(u,R):
	return R[u,:]

# Simple exponential discount
def disc(k):
	beta = 0.9
	return np.power(beta, k)

def gini(x):
	x = np.sort(x)
	n = x.shape[0]
	xbar = np.mean(x)
	#Calc Gini using unordered data (Damgaard & Weiner, Ecology 2000)
	absdif = 0
	for i in range(n):
		for j in range(n): absdif += abs(x[i]-x[j])
	G = absdif/(2*np.power(n,2)*xbar) * (n/(n)) # change to n/(n-1) for unbiased
	return G

def computeGinis(S, C):
	""" Gini diversity measure

	Based on: Kartik Hosanagar, Daniel Fleder (2008)

	"""
	GiniPerRec = {}
	S = S - C
	G1 = gini(np.sum(C,axis=0))
	G2 = gini(np.sum(S,axis=0))
	return G2 - G1


def metrics(M,Rec,ItemFeatures,dist,Mafter):
	U_ = M.shape[0]
	I_ = M.shape[1]
	Rtest = Mafter - M
	RecAsMatrix = np.zeros((U_, I_))
	for u in Rec.keys():
		RecAsMatrix[u,Rec[u]]=1
	s = time.time()
	(mEPC,sEPC) = EPC(Rec,RecAsMatrix,M,U_,Rtest)
	e = time.time()
	print("time:",e-s)
	#(mILD,sILD) = ILD(Rec,RecAsMatrix,M,U_,dist)
	#(mEFD,sEFD) = EFD(Rec,RecAsMatrix,M,U_,Rtest)
	s = time.time()
	(mEPD,sEPD) = EPD(Rec,RecAsMatrix,M,U_,Rtest,dist)
	e = time.time()
	print("time:",e-s)
	#(mEILD,sEILD) = EILD(Rec,RecAsMatrix,M,U_,Rtest,dist)
	return {"EPC" :  mEPC,
	"EPCstd" :  sEPC,
	"ILD": 0,
	"ILDstd": 0,
	"EFD": 0,
	"EFDstd": 0,  
	"EPD": mEPD,
	"EPDstd": mEPD,
	"EILD": 0,
	"EILDstd": 0}
	