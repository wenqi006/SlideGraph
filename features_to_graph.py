# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, Wenqi Lu, Fayyaz Minhas, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

import numpy as np
import os
from scipy.spatial import Delaunay, KDTree
from collections import defaultdict
import torch
from torch.autograd import Variable
from torch_geometric.data import Data
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
from sklearn.neighbors import KDTree as sKDTree
from tqdm import tqdm
import pickle
USE_CUDA = torch.cuda.is_available()

def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v

def toTensor(v,dtype = torch.float,requires_grad = True): 
    device = 'cuda:0'   
    return (Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad)).to(device)

def connectClusters(Cc,dthresh = 3000):
    tess = Delaunay(Cc)
    neighbors = defaultdict(set)
    for simplex in tess.simplices:
        for idx in simplex:
            other = set(simplex)
            other.remove(idx)
            neighbors[idx] = neighbors[idx].union(other)
    nx = neighbors    
    W = np.zeros((Cc.shape[0],Cc.shape[0]))
    for n in nx:
        nx[n] = np.array(list(nx[n]),dtype = np.int)
        nx[n] = nx[n][KDTree(Cc[nx[n],:]).query_ball_point(Cc[n],r = dthresh)]
        W[n,nx[n]] = 1.0
        W[nx[n],n] = 1.0        
    return W # neighbors of each cluster and an affinity matrix

def toGeometric(X,W,y,tt=0):    
    return Data(x=toTensor(X,requires_grad = False), edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),y=toTensor([y],dtype=torch.long,requires_grad = False))

if __name__ == '__main__':
    # similarity parameters
    lambda_d = 3e-3 
    lambda_f = 1.0e-3
    feature_path = './example' # load x, y coordinates and features of patches in each WSI
    output_path = './graphs'
    for filename in tqdm(os.listdir(feature_path)):
        print(filename)
        ofile = os.path.join(output_path, filename[:-4] + '.pkl')
        if os.path.isfile(ofile):
            continue
        label = int(1)
        if filename.endswith(".npz"):
            d = np.load(feature_path + '/' + filename, allow_pickle=True)
            x, y, F = d['x_patch'], d['y_patch'], d['feature']
            ridx = (np.max(F, axis=0) - np.min(F, axis=0)) > 1e-4 # remove feature which does not change
            F = F[:, ridx]
            C = np.asarray(np.vstack((x, y)).T, dtype=np.int)
            TC = sKDTree(C)
            I, D = TC.query_radius(C, r=6 / lambda_d, return_distance=True, sort_results=True)
            DX = np.zeros(int(C.shape[0] * (C.shape[0] - 1) / 2))
            idx = 0
            for i in range(C.shape[0] - 1):
                f = np.exp(-lambda_f * np.linalg.norm(F[i] - F[I[i]], axis=1))
                d = np.exp(-lambda_d * D[i])
                df = 1 - f * d
                dfi = np.ones(C.shape[0])
                dfi[I[i]] = df
                dfi = dfi[i + 1:]
                DX[idx:idx + len(dfi)] = dfi
                idx = idx + len(dfi)
            d = DX

            # %%
            lamda_h = 0.8. # Hierachical clustering distance threshold
            Z = hierarchy.linkage(d, method='average')
            clusters = fcluster(Z, lamda_h, criterion='distance')
            uc = list(set(clusters))
            C_cluster = []
            F_cluster = []
            for c in uc:
                idx = np.where(clusters == c)
                if C[idx, :].squeeze().size==2:
                    C_cluster.append(list(np.round(C[idx, :].squeeze())))
                    F_cluster.append(list(F[idx, :].squeeze()))
                else:
                    C_cluster.append(list(np.round(C[idx, :].squeeze().mean(axis=0))))
                    F_cluster.append(list(F[idx, :].squeeze().mean(axis=0)))
            C_cluster = np.array(C_cluster)
            F_cluster = np.array((F_cluster))

            W = connectClusters(C_cluster, dthresh=4000)
            G = toGeometric(F_cluster, W, y=label)
            G.coords = toTensor(C_cluster, requires_grad=False)

            with open(ofile, 'wb') as f:
                pickle.dump(G, f)

