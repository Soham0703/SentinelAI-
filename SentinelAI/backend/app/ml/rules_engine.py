from typing import Dict, List, Any


class RulesEngine:
    @staticmethod
    def evaluate(transaction: Dict[str, Any]) -> List[Dict[str, Any]]:
        rules_triggered = []
        amount = float(transaction.get('amount', 0.0))
        old_orig = float(transaction.get('oldbalanceOrg', 0.0))
        new_orig = float(transaction.get('newbalanceOrig', 0.0))
        old_dest = float(transaction.get('oldbalanceDest', 0.0))
        new_dest = float(transaction.get('newbalanceDest', 0.0))
        tx_type = str(transaction.get('type', 'TRANSFER')).upper()
        dest_name = str(transaction.get('nameDest', '')).upper()
        step = int(transaction.get('step', 12))
        hour = step % 24

        if amount >= 20000.0 and new_orig == 0.0 and old_orig > 0:
            rules_triggered.append({
                "rule_id": "RULE_001_TOTAL_DRAIN",
                "severity": "CRITICAL",
                "score_impact": 0.35,
                "description": f"Liquidated entire balance (${amount:,.2f}) to $0.00 in a single operation."
            })

        if 'MULE' in dest_name:
            rules_triggered.append({
                "rule_id": "RULE_002_MULE_HUB_MATCH",
                "severity": "CRITICAL",
                "score_impact": 0.40,
                "description": f"Destination counterparty '{dest_name}' flagged in active money mule intelligence list."
            })

        if tx_type in ['TRANSFER', 'CASH_OUT'] and old_dest == 0.0 and new_dest == 0.0 and amount > 5000.0:
            rules_triggered.append({
                "rule_id": "RULE_003_BALANCE_ANOMALY",
                "severity": "HIGH",
                "score_impact": 0.25,
                "description": "Destination account balance remained $0 despite incoming funds, suggesting rapid layered extraction."
            })

        if tx_type in ['TRANSFER', 'CASH_OUT'] and (hour >= 1 and hour <= 4) and amount > 10000.0:
            rules_triggered.append({
                "rule_id": "RULE_004_OFF_HOURS_TRANSFER",
                "severity": "MEDIUM",
                "score_impact": 0.15,
                "description": f"High-value transfer initiated during high-risk off-hours window ({hour:02d}:00 UTC)."
            })

        if 9500.0 <= amount <= 9999.0:
            rules_triggered.append({
                "rule_id": "RULE_005_STRUCTURING_FLAG",
                "severity": "HIGH",
                "score_impact": 0.25,
                "description": f"Transaction amount (${amount:,.2f}) borders the $10,000 regulatory reporting threshold (possible smurfing)."
            })

        return rules_triggered
