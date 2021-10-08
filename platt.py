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
class PlattScaling:
    def __init__(self):
        self.A = None
        self.B = None
    def fit(self,L,V):
        """
        Fit the sigmoid to the classifier scores V and labels L  using the Platt Method
        Input:  V array-like of classifier output scores
                L array like of classifier labels (+1/-1 pr +1/0)
        Output: Coefficients A and B for the sigmoid function
        """
        def mylog(v):
            if v==0:
                return -200
            else: 
                return np.log(v)
        out = np.array(V)
        L = np.array(L)
        assert len(V)==len(L)
        target = L==1
        prior1 = np.float(np.sum(target))
        prior0 = len(target)-prior1    
        A = 0
        B = np.log((prior0+1)/(prior1+1))
        self.A,self.B = A,B
        hiTarget = (prior1+1)/(prior1+2)
        loTarget = 1/(prior0+2)
        labda = 1e-3
        olderr = 1e300
        pp = np.ones(out.shape)*(prior1+1)/(prior0+prior1+2)
        T = np.zeros(target.shape)
        for it in range(1,100):
            a = 0
            b = 0
            c = 0
            d = 0
            e = 0
            for i in range(len(out)):
                if target[i]:
                    t = hiTarget
                    T[i] = t
                else:
                    t = loTarget
                    T[i] = t
                d1 = pp[i]-t
                d2 = pp[i]*(1-pp[i])
                a+=out[i]*out[i]*d2
                b+=d2
                c+=out[i]*d2
                d+=out[i]*d1
                e+=d1
            if (abs(d)<1e-9 and abs(e)<1e-9):
                break
            oldA = A
            oldB = B
            err = 0
            count = 0
            while 1:
                det = (a+labda)*(b+labda)-c*c
                if det == 0:
                    labda *= 10
                    continue
                A = oldA+ ((b+labda)*d-c*e)/det
                B = oldB+ ((a+labda)*e-c*d)/det
                self.A,self.B = A,B
                err = 0
                for i in range(len(out)):            
                    p = self.transform(out[i])
                    pp[i]=p
                    t = T[i]
                    err-=t*mylog(p)+(1-t)*mylog(1-p)
                if err<olderr*(1+1e-7):
                    labda *= 0.1
                    break
                labda*=10
                if labda>1e6:
                    break
                diff = err-olderr
                scale = 0.5*(err+olderr+1)
                if diff>-1e-3*scale and diff <1e-7*scale:
                    count+=1
                else:
                    count = 0
                olderr = err
                if count == 3:
                    break
        self.A,self.B = A,B
        return self
    def transform(self,V):       
        return 1/(1+np.exp(V*self.A+self.B))
    
    def fit_transform(self,L,V):
        return self.fit(L,V).transform(V)

    def __repr__(self):
        A,B = self.A,self.B
        return "Platt Scaling: "+f'A: {A}, B: {B}'


if __name__ == '__main__':

    V = 3*(2*np.random.rand(100)-1) #classifier output raw scores
    L = 2*((V+2*np.random.rand(len(V))-1)>0)-1 #Original binary labels
    pp = PlattScaling().fit_transform(L, V) #rescling-coefficients
    #print('A =',A,'B =',B)
    #pp = sigmoid(V,A,B)
    from sklearn.metrics import roc_auc_score
    print("Print Ranges:")
    
    print("Original:",np.min(V),np.max(V))
    print("Rescaled:",np.min(pp),np.max(pp))
    print("Calculate AUC-ROC (should not change):")
    print(roc_auc_score(L,pp))
    print(roc_auc_score(L,V))
