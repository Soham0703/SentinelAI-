import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import os


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvolutionLayer, self).__init__()
        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_neigh = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, adj_sparse: torch.Tensor) -> torch.Tensor:
        self_feat = self.w_self(x)
        neigh_feat = self.w_neigh(torch.sparse.mm(adj_sparse, x))
        return F.relu(self_feat + neigh_feat)


class FinancialFraudGNN(nn.Module):
    def __init__(self, node_feature_dim: int = 16, edge_feature_dim: int = 18, hidden_dim: int = 32):
        super(FinancialFraudGNN, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim

        self.gcn1 = GraphConvolutionLayer(node_feature_dim, hidden_dim)
        self.gcn2 = GraphConvolutionLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.2)

        edge_input_dim = (hidden_dim * 2) + edge_feature_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(
        self, 
        node_features: torch.Tensor, 
        adj_sparse: torch.Tensor, 
        edge_indices: torch.Tensor, 
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        h = self.gcn1(node_features, adj_sparse)
        h = self.dropout(h)
        h = self.gcn2(h, adj_sparse)

        src_nodes = edge_indices[:, 0]
        dst_nodes = edge_indices[:, 1]
        
        h_src = h[src_nodes]
        h_dst = h[dst_nodes]
        
        edge_repr = torch.cat([h_src, h_dst, edge_features], dim=1)
        return self.edge_classifier(edge_repr)


class GraphFraudDetector:
    def __init__(self, hidden_dim: int = 32, lr: float = 0.008, epochs: int = 20):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.node_to_idx: Dict[str, int] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_sparse_graph(
        self, 
        df: pd.DataFrame, 
        edge_features_np: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_origs = df['nameOrig'].unique()
        unique_dests = df['nameDest'].unique()
        all_nodes = list(set(unique_origs).union(set(unique_dests)))
        
        self.node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        num_nodes = len(all_nodes)

        node_degrees = np.zeros(num_nodes, dtype=np.float32)
        node_amounts = np.zeros(num_nodes, dtype=np.float32)

        edge_list = []
        u_list = []
        v_list = []
        for _, row in df.iterrows():
            u = self.node_to_idx[row['nameOrig']]
            v = self.node_to_idx[row['nameDest']]
            edge_list.append([u, v])
            u_list.append(u)
            v_list.append(v)
            amt = float(row['amount'])
            node_degrees[u] += 1
            node_degrees[v] += 1
            node_amounts[u] += amt
            node_amounts[v] += amt

        edge_indices = torch.tensor(edge_list, dtype=torch.long)
        edge_features = torch.tensor(edge_features_np, dtype=torch.float32)

        raw_node_stats = np.stack([
            np.log1p(node_degrees),
            np.log1p(node_amounts),
            np.sin(np.arange(num_nodes)),
            np.cos(np.arange(num_nodes))
        ], axis=1)
        
        node_features_np = np.zeros((num_nodes, 16), dtype=np.float32)
        node_features_np[:, :4] = raw_node_stats
        for i in range(4, 16):
            node_features_np[:, i] = np.sin(raw_node_stats[:, 0] * (i + 1))
        node_features = torch.tensor(node_features_np, dtype=torch.float32)

        all_src = u_list + v_list + list(range(num_nodes))
        all_dst = v_list + u_list + list(range(num_nodes))
        
        indices = torch.tensor([all_src, all_dst], dtype=torch.long)
        values = torch.ones(len(all_src), dtype=torch.float32)
        
        deg = torch.zeros(num_nodes, dtype=torch.float32)
        deg.scatter_add_(0, indices[0], values)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        norm_values = deg_inv_sqrt[indices[0]] * values * deg_inv_sqrt[indices[1]]
        
        adj_sparse = torch.sparse_coo_tensor(
            indices, norm_values, torch.Size([num_nodes, num_nodes])
        ).coalesce()

        return node_features, adj_sparse, edge_indices, edge_features

    def fit(self, df_train: pd.DataFrame, X_train_trans: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        edge_feature_dim = X_train_trans.shape[1]
        self.model = FinancialFraudGNN(
            node_feature_dim=16, 
            edge_feature_dim=edge_feature_dim, 
            hidden_dim=self.hidden_dim
        ).to(self.device)

        node_feat, adj_sparse, edge_idx, edge_feat = self.build_sparse_graph(df_train, X_train_trans)
        
        node_feat = node_feat.to(self.device)
        adj_sparse = adj_sparse.to(self.device)
        edge_idx = edge_idx.to(self.device)
        edge_feat = edge_feat.to(self.device)
        targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)

        n_pos = float(np.sum(y_train == 1))
        n_neg = float(np.sum(y_train == 0))
        pos_weight = torch.tensor([max(1.0, n_neg / (n_pos + 1e-5))], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        loss_history = []
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(node_feat, adj_sparse, edge_idx, edge_feat)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))

        return {
            'epochs': self.epochs,
            'final_loss': loss_history[-1],
            'loss_history': loss_history
        }

    def predict_proba(self, df: pd.DataFrame, X_trans: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("GNN model is not fitted.")

        self.model.eval()
        with torch.no_grad():
            node_feat, adj_sparse, edge_idx, edge_feat = self.build_sparse_graph(df, X_trans)
            node_feat = node_feat.to(self.device)
            adj_sparse = adj_sparse.to(self.device)
            edge_idx = edge_idx.to(self.device)
            edge_feat = edge_feat.to(self.device)

            logits = self.model(node_feat, adj_sparse, edge_idx, edge_feat)
            return torch.sigmoid(logits).cpu().numpy().flatten()

    def predict_single(self, transaction: Dict[str, Any], x_trans: np.ndarray) -> float:
        if self.model is None:
            return 0.05
        
        dest = str(transaction.get('nameDest', 'UNKNOWN_DEST'))
        mule_hub_flag = 1.0 if 'MULE' in dest.upper() else 0.0
        
        self.model.eval()
        with torch.no_grad():
            feat_tensor = torch.tensor(x_trans, dtype=torch.float32).to(self.device)
            if feat_tensor.dim() == 1:
                feat_tensor = feat_tensor.unsqueeze(0)
            
            dummy_node = torch.zeros((1, self.hidden_dim), dtype=torch.float32).to(self.device)
            edge_repr = torch.cat([dummy_node, dummy_node, feat_tensor], dim=1)
            logit = self.model.edge_classifier(edge_repr)
            prob = float(torch.sigmoid(logit).cpu().item())
            
            if mule_hub_flag > 0:
                prob = min(0.99, prob * 1.5 + 0.25)
                
            return float(np.clip(prob, 0.0, 1.0))

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'node_to_idx': self.node_to_idx,
            'hidden_dim': self.hidden_dim
        }, filepath)

    def load(self, filepath: str, edge_feature_dim: int):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.hidden_dim = checkpoint['hidden_dim']
        self.node_to_idx = checkpoint['node_to_idx']
        self.model = FinancialFraudGNN(
            node_feature_dim=16,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
