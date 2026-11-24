from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, List, Any

from app.db.database import get_db
from app.db.models import TransactionRecord
from app.db.schemas import GraphNetworkResponse, GraphNode, GraphEdge

router = APIRouter(prefix="", tags=["Graph"])


@router.get("/graph/subgraph", response_model=GraphNetworkResponse)
def get_graph_subgraph(
    limit: int = Query(60, ge=10, le=200),
    mule_focus: bool = Query(True),
    db: Session = Depends(get_db)
):
    query = db.query(TransactionRecord)
    if mule_focus:
        records = query.filter(TransactionRecord.riskScore > 30.0).order_by(TransactionRecord.riskScore.desc()).limit(limit).all()
        if len(records) < 15:
            records = query.order_by(TransactionRecord.id.desc()).limit(limit).all()
    else:
        records = query.order_by(TransactionRecord.id.desc()).limit(limit).all()

    nodes_dict: Dict[str, Dict[str, Any]] = {}
    edges_list: List[GraphEdge] = []

    for tx in records:
        orig = tx.nameOrig
        dest = tx.nameDest
        amt = float(tx.amount)
        risk = float(tx.riskScore)
        is_fraud = bool(tx.predictedFraud == 1)

        dest_type = "DEST_MULE" if "MULE" in dest.upper() else ("DEST_MERCHANT" if "M" in dest.upper() else "DEST_ACCOUNT")

        if orig not in nodes_dict:
            nodes_dict[orig] = {
                "id": orig,
                "label": orig,
                "type": "ORIGIN",
                "risk_score": risk,
                "degree": 1
            }
        else:
            nodes_dict[orig]["degree"] += 1
            nodes_dict[orig]["risk_score"] = max(nodes_dict[orig]["risk_score"], risk)

        if dest not in nodes_dict:
            nodes_dict[dest] = {
                "id": dest,
                "label": dest,
                "type": dest_type,
                "risk_score": risk,
                "degree": 1
            }
        else:
            nodes_dict[dest]["degree"] += 1
            nodes_dict[dest]["risk_score"] = max(nodes_dict[dest]["risk_score"], risk)

        edges_list.append(GraphEdge(
            source=orig,
            target=dest,
            amount=amt,
            type=tx.type,
            is_fraud=is_fraud,
            risk_score=risk
        ))

    nodes_list = [
        GraphNode(
            id=v["id"],
            label=v["label"],
            type=v["type"],
            risk_score=round(v["risk_score"], 2),
            degree=v["degree"]
        )
        for v in nodes_dict.values()
    ]

    mule_count = sum(1 for n in nodes_list if n.type == "DEST_MULE")
    high_risk_edges = sum(1 for e in edges_list if e.is_fraud)

    return {
        "nodes": nodes_list,
        "edges": edges_list,
        "stats": {
            "node_count": len(nodes_list),
            "edge_count": len(edges_list),
            "identified_mules": mule_count,
            "compromised_flows": high_risk_edges
        }
    }
