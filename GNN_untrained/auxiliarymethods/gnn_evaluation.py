import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from tqdm import tqdm


from auxiliarymethods.danutils import *

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


# One training epoch for GNN model.
def train(train_loader, model, optimizer, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()


# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


# 10-CV for GNN training and hyperparameter selection.
def gnn_evaluation(gnn, ds_name, layers, hidden, max_num_epochs=200, batch_size=128, start_lr=0.01, min_lr = 0.000001, factor=0.5, patience=5,
                       num_repetitions=10, all_std=True, untrain=False):
    #reproducibility
    torch.manual_seed(0)
    # Load dataset and shuffle.
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', ds_name)
    dataset = TUDataset(path, name=ds_name).shuffle()

    # Constant node features
    if dataset[0].x is None:
        printd('no node features')
        dataset.transform = T.Constant(value=1)
        # max_degree = 0
        # degs = []
        # for data in dataset:
        #     degs += [degree(data.edge_index[0], dtype=torch.long)]
        #     max_degree = max(max_degree, degs[-1].max().item())

        # if max_degree < 1000:
        #     dataset.transform = T.OneHotDegree(max_degree)
        # else:
        #     deg = torch.cat(degs, dim=0).to(torch.float)
        #     mean, std = deg.mean().item(), deg.std().item()
        #     dataset.transform = NormalizedDegree(mean, std)

    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_accuracies_all = []
    test_accuracies_complete = []

    printd(f"num_repetitions = {num_repetitions}")
    for i in tqdm(range(num_repetitions)):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True, random_state=i)
        
        #dataset.shuffle()

        splits = kf.split(list(range(len(dataset))))
        for train_index, test_index in tqdm(splits):
            #! IT LOOKS LIKE CROSS VALIDATION ON TOP OF CROSS VALIDATION. WHY DO WE SPLIT INTO TRAIN,VAL,TEST MULTIPLE TIMES AND NOT ONCE?
            # Sample 10% split from training split for validation. 
            #? dan: kf.split() returns an array of 10 splits of the data into (train, test) indices. every time we do the split, we get a new array to iterate on. 

            train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=i)
            #? dan: every time we want to do across validation
            best_val_acc = 0.0
            best_test = 0.0
            # Split data.
            train_dataset = dataset[train_index.tolist()]
            val_dataset = dataset[val_index.tolist()]
            test_dataset = dataset[test_index.tolist()]

            # Prepare batching.
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #reproducible

            # Collect val. and test acc. over all hyperparameter combinations.
            for l in layers:
                for h in hidden:
                    # Setup model.
                    #? dan: so we train one model for every l and h
                    model = gnn(dataset, l, h, untrain).to(device)
                    #? dan: for the untrained networks, the reset_parameters function sets requires_grad = False 
                    model.reset_parameters()

                    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                           factor=factor, patience=patience,
                                                                           min_lr=0.0000001)
                    times = []
                    printd(f"training epochs")
                    for epoch in tqdm(range(1, max_num_epochs + 1)):
                        lr = scheduler.optimizer.param_groups[0]['lr']
                        #count training time per epoch
                        
                        #? Timing the training function
                        if device.type == 'cuda':
                            torch.cuda.synchronize() 
                        start_time = time.time()
                        
                        train(train_loader, model, optimizer, device)
                        

                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        end_time = time.time()
                        elapsed = end_time - start_time
                        times.append(elapsed)
                        

                        val_acc = test(val_loader, model, device)
                        scheduler.step(val_acc)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_test = test(test_loader, model, device) * 100.0

                        # Break if learning rate is smaller 10**-6.
                        if lr < min_lr: 
                            break           
                                        
                    mean_time = np.array(times).mean()
                    std_time = np.array(times).std()
                    printd(f"time for training: {mean_time} +- {std_time}")

            test_accuracies.append(best_test)

            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std(), np.array(times).mean(), np.array(times).std())
    else:
        return (np.array(test_accuracies_all).mean(),  np.array(times).mean(), np.array(times).std())

