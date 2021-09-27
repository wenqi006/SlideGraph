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
    device = 'cuda:1'   
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
    feature_path = './hover_morphological_features' # load x, y coordinates and features of patches in each WSI
    hover_count_path = ''
    output_path = './graphs'
    for filename in tqdm(os.listdir(feature_path)):
        print(filename)
        ofile = os.path.join(output_path, filename[:23] + '.pkl')
        if os.path.isfile(ofile):
            continue
        label = int(1)

        if filename.endswith(".npz"):
            d = np.load(feature_path + '/' + filename, allow_pickle=True)
            x, y, F = d['x_patch'], d['y_patch'], d['hover_mophological_feature']
            ###############################
            # extract features of morphology
            F_patch = []
            index_empty_array = []
            for no_patch in range(len(F)):
                cell_info_per_patch = F[no_patch]
                if cell_info_per_patch == []:
                    index_empty_array.append(int(no_patch))
                    continue
                mean_per_patch = np.mean(np.array(cell_info_per_patch)[:,:-2], axis=0)
                std_per_patch = np.std(np.array(cell_info_per_patch)[:,:-2], axis=0)
                arr = np.concatenate((mean_per_patch, std_per_patch))
                F_patch.append(arr)
            ###############################
            x = np.delete(x, np.array(index_empty_array), axis=0)
            y = np.delete(y, np.array(index_empty_array), axis=0)
            F = np.array(F_patch)
            C = np.asarray(np.vstack((x, y)).T, dtype=np.int)

            lambda_d = 3e-3
            lambda_f = 1.0e-3

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
            lamda_h = 0.8
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
