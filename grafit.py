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

from utils import *

#%% Graph Fitting
class GraphFit:
    """
    A Pytorch implementation of "Fittng a graph to vector data"
    @inproceedings{Daitch:2009:FGV:1553374.1553400,
     author = {Daitch, Samuel I. and Kelner, Jonathan A. and Spielman, Daniel A.},
     title = {Fitting a Graph to Vector Data},
     booktitle = {Proceedings of the 26th Annual International Conference on Machine Learning},
     series = {ICML '09},
     year = {2009},
     isbn = {978-1-60558-516-1},
     location = {Montreal, Quebec, Canada},
     pages = {201--208},
     numpages = {8},
     url = {http://doi.acm.org/10.1145/1553374.1553400},
     doi = {10.1145/1553374.1553400},
     acmid = {1553400},
     publisher = {ACM},
     address = {New York, NY, USA},
    }     
    
    Solves: min_w \sum_i {d_i x_i - \sum_j w_{i,j}x_j}
    such that:
        \sum_i max{0,1-d_i}^2 \le \alpha n    
    """    
    def __init__(self,n,d):          
        self.n,self.d = n,d
        self.W = cuda(Variable(torch.rand((n,n)).float())).requires_grad_()
        self.history = []
    def fit(self,X,lr=1e-2,epochs=500):
        X = toTensor(X,requires_grad=False)
        self.X = X              
        optimizer = optim.Adam([self.W], lr=lr)        
        alpha = 1.0      
        zero = toTensor([0])        
        for epochs in range(epochs):
            L,D = self.getLaplacianDegree()                 
            loss = torch.norm(L@X)+alpha*torch.sum(torch.max(zero,1-D)**2)
            
            self.history.append(loss.item())
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
        
        return self
            
    def getLaplacianDegree(self):
        W = self.getW()            
        L = -W
        D = torch.sum(W,dim=0)
        L.diagonal(dim1=-2, dim2=-1).copy_(D)  
        return L,D
            
    def getW(self):
        """
        Gets adjacency matrix for the graph
        """
        Z = (torch.transpose(self.W, 0, 1)+self.W)**2
        Z.fill_diagonal_(0)
        return Z
    
