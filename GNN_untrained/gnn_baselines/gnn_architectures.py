import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, GraphConv, GCNConv, global_add_pool, global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset

from torch_geometric.utils import degree #for graph size normalization in the meanpool achitectures

# Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py.
class GIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, untrain):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Sequential(
            Linear(dataset.num_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ),
            train_eps=True)

        self.untrain = untrain

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                    train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        if self.untrain:
            for param in self.conv1.parameters():
                param.requires_grad = False

            for conv in self.convs:
                for param in conv.parameters():
                    param.requires_grad = False

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GIN_meanpool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, untrain):
        super(GIN_meanpool, self).__init__()
        self.conv1 = GINConv(Sequential(
            Linear(dataset.num_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ),
            train_eps=True)

        self.untrain = untrain

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                    train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        if self.untrain:
            for param in self.conv1.parameters():
                param.requires_grad = False

            for conv in self.convs:
                for param in conv.parameters():
                    param.requires_grad = False

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        inv_graph_size = degree(batch, dtype=x.dtype).pow(-1)
        x = x * inv_graph_size.index_select(0, batch).view(-1, 1)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = x * inv_graph_size.index_select(0, batch).view(-1, 1)

        x = global_mean_pool(x, batch) #use mean pooling

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GINEConv(MessagePassing):
    def __init__(self, edge_dim, dim_init, dim):
        super(GINEConv, self).__init__(aggr="add")

        self.edge_encoder = Sequential(Linear(edge_dim, dim_init), ReLU(), Linear(dim_init, dim_init), ReLU(),
                                       BN(dim_init))
        self.mlp = Sequential(Linear(dim_init, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.initial_eps = 0

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        reset(self.edge_encoder)
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)


class GINE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, untrain):
        super(GINE, self).__init__()
        self.conv1 = GINEConv(dataset.num_edge_features, dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GINEConv(dataset.num_edge_features, hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.untrain = untrain

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        if self.untrain:
            for param in self.conv1.parameters():
                param.requires_grad = False

            for conv in self.convs:
                for param in conv.parameters():
                    param.requires_grad = False

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GC(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, untrain):
        super(GC, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GraphConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.untrain = untrain

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        if self.untrain:
            #? DAN: i think that untrain is because i don't really train the networks
            for param in self.conv1.parameters():
                param.requires_grad = False

            for conv in self.convs:
                for param in conv.parameters():
                    param.requires_grad = False

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)

        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GC_meanpool(torch.nn.Module):
    #? DAN: this is what we are using. pretty standard stuff, except with the GraphConv instead of what you did.
    def __init__(self, dataset, num_layers, hidden, untrain):
        super(GC_meanpool, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GraphConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.untrain = untrain

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        if self.untrain:
            for param in self.conv1.parameters():
                param.requires_grad = False

            for conv in self.convs:
                for param in conv.parameters():
                    param.requires_grad = False

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        inv_graph_size = degree(batch, dtype=x.dtype).pow(-1)
        #! shouldn't the normalization come before the conv?
        x = x * inv_graph_size.index_select(0, batch).view(-1, 1)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = x * inv_graph_size.index_select(0, batch).view(-1, 1)

        x = global_mean_pool(x, batch) #change it to mean pooling

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, untrain):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.untrain = untrain

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        if self.untrain:
            for param in self.conv1.parameters():
                param.requires_grad = False

            for conv in self.convs:
                for param in conv.parameters():
                    param.requires_grad = False

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
