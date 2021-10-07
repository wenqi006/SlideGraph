from glob import glob
import os
import pandas as pd
import numpy as np
import pickle
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.model_selection import StratifiedKFold, train_test_split

def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)

def toTensor(v, dtype=torch.float, requires_grad=True):
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))

if __name__ == '__main__':

    learning_rate = 0.001
    weight_decay = 0.0001
    epochs = 300 # Total number of epochs
    split_fold = 5 # Stratified cross validation
    scheduler = None

    # Load clinical data
    clin_path = './CLINI.csv'
    clin_data = pd.read_csv(clin_path)  # path to clinical file
    Patient_ID = np.array(clin_data['PATIENT'])
    Receptor_status = clin_data['HER2FinalStatus']

    # Load graphs
    bdir = './graphs'  # path to graphs
    graphlist = glob(os.path.join(bdir, "*.pkl"))
    GN = []
    device = 'cuda:0'
    dataset = []
    for graph in tqdm(graphlist):
        G = pickleLoad(graph)
        G = G.to(device)
        # Find the index of the patient ID of the current graph inside of the clinical file 
        idx = np.nonzero(Patient_ID == os.path.split(graph)[-1].split('_')[0][:12])[0]
        # If no matched patient ID, go to the next graph
        if len(idx) == 0:
            continue
        status = list(Receptor_status[idx])[0]
        # Only consider patients whose receptor status is positive or negative
        if status not in ['Positive', 'Negative']:
            continue
        G.y = toTensor([int(status == 'Positive')], dtype=torch.long, requires_grad=False)
        GN.append(G.x)
        dataset.append(G)

    # Normalise features
    GN = torch.cat(GN)
    Gmean, Gstd = torch.mean(GN, dim=0), torch.std(GN, dim=0)
    for G in dataset:
        G.x = (G.x - Gmean) / Gstd

    Y = [float(G.y) for G in dataset]

    skf = StratifiedKFold(n_splits=split_fold, shuffle=False)  # Stratified cross validation
    Vacc, Tacc, Vapr, Tapr, Test_ROC_overall, Test_PR_overall = [], [], [], [], [],  [] # Intialise outputs

    Fdata = []
    for trvi, test in skf.split(dataset, Y):
        train, valid = train_test_split(trvi, test_size=0.10, shuffle=True,
                                        stratify=np.array(Y)[trvi])  # 10% for validation and 90% for training 
        sampler = StratifiedSampler(class_vector=torch.from_numpy(np.array(Y)[train]), batch_size=8)
        train_dataset = [dataset[i] for i in train]
        tr_loader = DataLoader(train_dataset, batch_sampler=sampler)
        valid_dataset = [dataset[i] for i in valid]
        v_loader = DataLoader(valid_dataset, shuffle=False)
        test_dataset = [dataset[i] for i in test]
        tt_loader = DataLoader(test_dataset, shuffle=False)

        model = GNN(dim_features=dataset[0].x.shape[1], dim_target=1, layers=[16, 16, 8], dropout=0.5, pooling='mean',
                    conv='EdgeConv', aggr='max') # model GNN architecture

        net = NetWrapper(model, loss_function=None, device=device)
        model = model.to(device=net.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_model, train_loss, train_acc, val_loss, val_acc, tt_loss, tt_acc, val_pr, test_pr = net.train(
            train_loader=tr_loader,
            max_epochs=epochs,
            optimizer=optimizer, 
            scheduler=scheduler,
            clipping=None,
            validation_loader=v_loader,
            test_loader=tt_loader,
            early_stopping=30,
            return_best=False,
            log_every=10)
        Fdata.append((best_model, test_dataset, valid_dataset))
        Vacc.append(val_acc)
        Tacc.append(tt_acc)
        Vapr.append(val_pr)
        Tapr.append(test_pr)
        print("\nfold complete", len(Vacc), train_acc, val_acc, tt_acc, val_pr, test_pr)

    # Averaged results of 5 folds
    print("avg Valid AUC=", np.mean(Vacc), "+/-", np.std(Vacc))
    print("avg Test AUC=", np.mean(Tacc), "+/-", np.std(Tacc))
    print("avg Valid PR=", np.mean(Vapr), "+/-", np.std(Vapr))
    print("avg Test PR=", np.mean(Tapr), "+/-", np.std(Tapr))

    # Use top 10 models in each fold and re-calculate the averaged results of 5 folds
    auroc = []
    aupr = []
    for idx in range(len(Fdata)):
        Q, test_dataset, valid_dataset = Fdata[idx]
        zz, yy = EnsembleDecisionScoring(Q, train_dataset, test_dataset, device=net.device, k=10)
        auroc.append(roc_auc_score(yy, zz))
        aupr.append(average_precision_score(yy, zz))

    print("avg Test AUC overall=", np.mean(auroc), "+/-", np.std(auroc))
    print("avg Test PR overall=", np.mean(bb), "+/-", np.std(aupr))
