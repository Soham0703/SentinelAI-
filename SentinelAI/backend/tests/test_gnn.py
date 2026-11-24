import pytest
import torch
import numpy as np
import pandas as pd
from app.ml.gnn_model import GraphFraudDetector, FinancialFraudGNN


def test_gnn_architecture():
    model = FinancialFraudGNN(node_feature_dim=16, edge_feature_dim=18, hidden_dim=32)
    assert model.hidden_dim == 32
    assert isinstance(model.edge_classifier, torch.nn.Sequential)


def test_sparse_graph_construction():
    detector = GraphFraudDetector()
    df = pd.DataFrame([
        {"nameOrig": "C1", "nameDest": "MULE_01", "amount": 1000.0},
        {"nameOrig": "C2", "nameDest": "MULE_01", "amount": 2000.0},
        {"nameOrig": "C3", "nameDest": "M1", "amount": 50.0}
    ])
    edge_feats = np.zeros((3, 18), dtype=np.float32)

    node_feats, adj_sparse, edge_idx, edge_feats_t = detector.build_sparse_graph(df, edge_feats)

    assert node_feats.shape[0] == len(detector.node_to_idx)
    assert adj_sparse.is_sparse is True
    assert edge_idx.shape == (3, 2)
    assert edge_feats_t.shape == (3, 18)
