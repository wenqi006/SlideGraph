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

import torch
from platt import PlattScaling
from utils import *
from torch.utils.data import Sampler
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix

class StratifiedSampler(Sampler):
    """Stratified Sampling
         return a stratified batch
    """
    def __init__(self, class_vector, batch_size = 10):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        """
        self.batch_size = batch_size
        self.n_splits = int(class_vector.size(0) / self.batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        import numpy as np
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits= self.n_splits,shuffle=True)
        YY = self.class_vector.numpy()
        idx = np.arange(len(YY))
        return [tidx for _,tidx in skf.split(idx,YY)] #return array of arrays of indices in each batch

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

def calc_roc_auc(target, prediction):    
    return roc_auc_score(toNumpy(target),toNumpy(prediction[:,-1]))

def calc_pr(target, prediction):
    return average_precision_score(toNumpy(target), toNumpy(prediction[:, -1]))

#%% Graph Neural Network 
class GNN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, layers=[6,6],pooling='max',dropout = 0.0,conv='GINConv',gembed=False,**kwargs):
        """
        Parameters
        ----------
        dim_features : TYPE Int
            DESCRIPTION. Number of features of each node
        dim_target : TYPE Int
            DESCRIPTION. Number of outputs
        layers : TYPE, optional List of number of nodes in each layer
            DESCRIPTION. The default is [6,6].
        pooling : TYPE, optional
            DESCRIPTION. The default is 'max'.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.0.
        conv : TYPE, optional Layer type string {'GINConv','EdgeConv'} supported
            DESCRIPTION. The default is 'GINConv'.
        gembed : TYPE, optional Graph Embedding
            DESCRIPTION. The default is False. Pool node scores or pool node features
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(GNN, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.pooling = {'max':global_max_pool,'mean':global_mean_pool,'add':global_add_pool}[pooling]
        self.gembed = gembed #if True then learn graph embedding for final classification (classify pooled node features) otherwise pool node decision scores

        #train_eps = True#config['train_eps']

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
                
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.linears.append(Linear(out_emb_dim, dim_target))                
                if conv=='GINConv':
                    subnet = Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                    self.nns.append(subnet)
                    self.convs.append(GINConv(self.nns[-1], **kwargs))  # Eq. 4.2 eps=100, train_eps=False
                elif conv=='EdgeConv':
                    subnet = Sequential(Linear(2*input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                    self.nns.append(subnet)                    
                    self.convs.append(EdgeConv(self.nns[-1],**kwargs))#DynamicEdgeConv#EdgeConv                aggr='mean'

                else:
                    raise NotImplementedError  
                    
        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        out = 0
        pooling = self.pooling
        Z = 0
        for layer in range(self.no_layers):            
            if layer == 0:
                x = self.first_h(x)
                z = self.linears[layer](x)
                Z+=z
                dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                out += dout
            else:
                x = self.convs[layer-1](x,edge_index)
                if not self.gembed:
                    z = self.linears[layer](x)
                    Z+=z
                    dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                else:
                    dout = F.dropout(self.linears[layer](pooling(x, batch)), p=self.dropout, training=self.training)
                out += dout
        return out,Z,x
    
#%% Wrapper for neetwork training   
    
def decision_function(model,loader,device='cpu',outOnly=True,returnNumpy=False): 
    """
    generate prediction score for a given model

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    loader : TYPE Dataset or dataloader
        DESCRIPTION.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    outOnly : TYPE, optional 
        DESCRIPTION. The default is True. Only return the prediction scores.
    returnNumpy : TYPE, optional
        DESCRIPTION. The default is False. Return numpy array or ttensor

    Returns
    -------
    Z : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    ZXn : TYPE
        DESCRIPTION. Empty unless outOnly is False

    """
    if type(loader) is not DataLoader: #if data is given
        loader = DataLoader(loader)
    if type(device)==type(''):
        device = torch.device(device)
    ZXn = []    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            output,zn,xn = model(data)
            if returnNumpy:
                zn,xn = toNumpy(zn),toNumpy(xn)
            if not outOnly:
                ZXn.append((zn,xn))
            if i == 0:
                Z = output
                Y = data.y
            else:
                Z = torch.cat((Z, output))
                Y = torch.cat((Y, data.y))
    if returnNumpy:
        Z,Y = toNumpy(Z),toNumpy(Y)
    
    return Z,Y,ZXn

from platt import PlattScaling
def EnsembleDecisionScoring(Q,train_dataset,test_dataset,device='cpu',k=None):
    """
    Generate prediction scores from an ensemble of models 
    First scales all prediction scores to the same range and then bags them

    Parameters
    ----------
    Q : TYPE reverse deque or list or tuple
        DESCRIPTION.  containing models or output of train function
    train_dataset : TYPE dataset or dataloader 
        DESCRIPTION.
    test_dataset : TYPE dataset or dataloader 
        DESCRIPTION. shuffle must be false
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    k : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    Z : Numpy array
        DESCRIPTION. Scores
    yy : Numpy array
        DESCRIPTION. Labels

    """
    
    Z = 0
    if k is None: k = len(Q)
    for i,mdl in enumerate(Q):            
        if type(mdl) in [tuple,list]:            mdl = mdl[0]
        zz,yy,_ = decision_function(mdl,train_dataset,device=device)            
        mdl.rescaler = PlattScaling().fit(toNumpy(yy),toNumpy(zz))
        zz,yy,_ = decision_function(mdl,test_dataset,device=device)
        zz,yy = mdl.rescaler.transform(toNumpy(zz)).ravel(),toNumpy(yy)
        Z+=zz
        if i+1==k: break
    Z=Z/k
    return  Z,yy
#%%   
class NetWrapper:
    def __init__(self, model, loss_function, device='cpu', classification=True):
        self.model = model
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.classification = classification
    def _pair_train(self,train_loader,optimizer,clipping = None):
        """
        Performs pairwise comparisons with ranking loss
        """
        
        model = self.model.to(self.device)
        model.train()
        loss_all = 0
        acc_all = 0
        assert self.classification
        for data in train_loader:
            
            data = data.to(self.device)
            
            optimizer.zero_grad()
            output,_,_ = model(data)
            #import pdb; pdb.set_trace()
            # Can add contrastive loss if reqd
            #import pdb; pdb.set_trace()
            y = data.y
            loss =0
            c = 0
            #z = Variable(torch.from_numpy(np.array(0))).type(torch.FloatTensor)
            z = toTensor([0])  
            for i in range(len(y)-1):
                for j in range(i+1,len(y)):
                    if y[i]!=y[j]:
                        c+=1
                        dz = output[i,-1]-output[j,-1]
                        dy = y[i]-y[j]                        
                        loss+=torch.max(z, 1.0-dy*dz)
                        #loss+=lossfun(zi,zj,dy)
            loss=loss/c

            acc = loss
            loss.backward()

            try:
                num_graphs = data.num_graphs
            except TypeError:
                num_graphs = data.adj.size(0)

            loss_all += loss.item() * num_graphs
            acc_all += acc.item() * num_graphs
      

            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()

        return acc_all / len(train_loader.dataset), loss_all / len(train_loader.dataset)
    
    def classify_graphs(self, loader):
        Z,Y,_ = decision_function(self.model,loader,device=self.device)
        if not isinstance(Z, tuple):
            Z = (Z,)
        #loss, acc = self.loss_fun(Y, *Z)
        loss = 0
        auc_val = calc_roc_auc(Y, *Z)
        pr = calc_pr(Y, *Z)
        return auc_val, loss, pr
        
    def train(self, train_loader, max_epochs=100, optimizer=torch.optim.Adam, scheduler=None, clipping=None,
              validation_loader=None, test_loader=None, early_stopping=100, return_best = True, log_every=0):
        """
        

        Parameters
        ----------
        train_loader : TYPE
            Training data loader.
        max_epochs : TYPE, optional
            DESCRIPTION. The default is 100.
        optimizer : TYPE, optional
            DESCRIPTION. The default is torch.optim.Adam.
        scheduler : TYPE, optional
            DESCRIPTION. The default is None.
        clipping : TYPE, optional
            DESCRIPTION. The default is None.
        validation_loader : TYPE, optional
            DESCRIPTION. The default is None.
        test_loader : TYPE, optional
            DESCRIPTION. The default is None.
        early_stopping : TYPE, optional
            Patience  parameter. The default is 100.
        return_best : TYPE, optional
            Return the models that give best validation performance. The default is True.
        log_every : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        Q : TYPE: (reversed) deque of tuples (model,val_acc,test_acc)
            DESCRIPTION. contains the last k models together with val and test acc
        train_loss : TYPE
            DESCRIPTION.
        train_acc : TYPE
            DESCRIPTION.
        val_loss : TYPE
            DESCRIPTION.
        val_acc : TYPE
            DESCRIPTION.
        test_loss : TYPE
            DESCRIPTION.
        test_acc : TYPE
            DESCRIPTION.

        """
        
        from collections import deque
        Q = deque(maxlen=10) # queue the last 5 models
        return_best = return_best and validation_loader is not None 
        val_loss, val_acc = -1, -1
        best_val_acc,test_acc_at_best_val_acc,val_pr_at_best_val_acc,test_pr_at_best_val_acc = -1,-1,-1,-1
        test_loss, test_acc = None, None
        time_per_epoch = []
        self.history = []   
        patience = early_stopping
        best_epoch = np.inf
        iterator = tqdm(range(1, max_epochs+1))        
        for epoch in iterator:
            updated = False

            if scheduler is not None:
                scheduler.step(epoch)
            start = time.time()
            
            train_acc, train_loss = self._pair_train(train_loader, optimizer, clipping)
            
            end = time.time() - start
            time_per_epoch.append(end)    
            if validation_loader is not None: 
                val_acc, val_loss, val_pr = self.classify_graphs(validation_loader)
            if test_loader is not None:
                test_acc, test_loss, test_pr = self.classify_graphs(test_loader)
            if val_acc>best_val_acc:
                best_val_acc = val_acc                
                test_acc_at_best_val_acc = test_acc
                val_pr_at_best_val_acc = val_pr
                test_pr_at_best_val_acc = test_pr
                best_epoch = epoch
                updated = True
                if return_best:
                    best_model = deepcopy(self.model)
                    Q.append((best_model,best_val_acc,test_acc_at_best_val_acc,val_pr_at_best_val_acc,test_pr_at_best_val_acc))

                if False: #or 
                    from vis import showGraphDataset,getVisData                
                    fig = showGraphDataset(getVisData(validation_loader,best_model,self.device,showNodeScore=False))
                    plt.savefig(f'./figout/{epoch}.jpg')
                    plt.close()
                    
            if not return_best:                   
                Q.append((deepcopy(self.model),val_acc,test_acc,val_pr,test_pr))
                   
            showresults = False
            if log_every==0: # show only if validation results improve
                showresults = updated
            elif (epoch-1) % log_every == 0:   
                showresults = True
                
            if showresults:                
                # msg = f'Epoch: {epoch}, TR loss: {train_loss} TR acc: {train_acc}, VL loss: {val_loss} VL acc: {val_acc} ' \
                #     f'TE loss: {test_loss} TE acc: {test_acc}'
                msg = f'Epoch: {epoch}, TR loss: {train_loss} TR perf: {train_acc}, VL perf: {val_acc} ' \
                    f'TE perf: {test_acc}, Best: VL perf: {best_val_acc} TE perf: {test_acc_at_best_val_acc} VL pr: {val_pr_at_best_val_acc} TE pr: {test_pr_at_best_val_acc}'
                tqdm.write('\n'+msg)                   
                self.history.append(train_loss)
                
            if epoch-best_epoch>patience: 
                iterator.close()
                break
            
        if return_best:
            val_acc = best_val_acc
            test_acc = test_acc_at_best_val_acc
            val_pr = val_pr_at_best_val_acc
            test_pr = test_pr_at_best_val_acc

        Q.reverse()    
        return Q,train_loss, train_acc, val_loss, np.round(val_acc, 2), test_loss, np.round(test_acc, 2), val_pr, test_pr
