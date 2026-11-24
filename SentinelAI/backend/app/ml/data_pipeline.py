import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os


class FinancialFraudDataPipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = []
        self.numeric_features = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 
            'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig', 
            'errorBalanceDest', 'amount_to_oldbalance_ratio', 
            'orig_velocity_1h', 'dest_velocity_1h', 
            'hour_sin', 'hour_cos', 'is_zero_balance_drain'
        ]
        self.categorical_features = ['type']

    def generate_benchmark_dataset(self, n_samples: int = 25000, fraud_ratio: float = 0.02) -> pd.DataFrame:
        np.random.seed(self.random_state)
        n_fraud = int(n_samples * fraud_ratio)
        n_legit = n_samples - n_fraud

        legit_types = np.random.choice(
            ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'],
            size=n_legit,
            p=[0.35, 0.15, 0.30, 0.05, 0.15]
        )
        
        legit_amounts = np.random.lognormal(mean=5.5, sigma=1.3, size=n_legit)
        legit_amounts = np.round(np.clip(legit_amounts, 5.0, 75000.0), 2)
        
        legit_oldbalance_orig = np.random.lognormal(mean=7.0, sigma=1.7, size=n_legit)
        legit_oldbalance_orig = np.round(np.maximum(legit_oldbalance_orig, legit_amounts * 0.8), 2)
        legit_newbalance_orig = np.round(np.maximum(0.0, legit_oldbalance_orig - legit_amounts), 2)

        legit_oldbalance_dest = np.random.lognormal(mean=6.5, sigma=1.8, size=n_legit)
        legit_oldbalance_dest = np.round(legit_oldbalance_dest, 2)
        
        dest_noise_mask = np.random.rand(n_legit) > 0.3
        legit_newbalance_dest = legit_oldbalance_dest.copy()
        legit_newbalance_dest[dest_noise_mask] = np.round(legit_oldbalance_dest[dest_noise_mask] + legit_amounts[dest_noise_mask], 2)

        legit_steps = np.random.randint(1, 721, size=n_legit)
        legit_origs = [f"C{np.random.randint(10000, 99999)}" for _ in range(n_legit)]
        
        popular_merchants = [f"M_HUB_{i:02d}" for i in range(1, 40)]
        legit_dests = []
        for t in legit_types:
            if t == 'PAYMENT' and np.random.rand() > 0.4:
                legit_dests.append(np.random.choice(popular_merchants))
            else:
                legit_dests.append(f"C{np.random.randint(10000, 99999)}")

        legit_df = pd.DataFrame({
            'step': legit_steps,
            'type': legit_types,
            'amount': legit_amounts,
            'nameOrig': legit_origs,
            'oldbalanceOrg': legit_oldbalance_orig,
            'newbalanceOrig': legit_newbalance_orig,
            'nameDest': legit_dests,
            'oldbalanceDest': legit_oldbalance_dest,
            'newbalanceDest': legit_newbalance_dest,
            'isFraud': 0
        })

        fraud_types = np.random.choice(['TRANSFER', 'CASH_OUT', 'PAYMENT'], size=n_fraud, p=[0.48, 0.46, 0.06])
        fraud_amounts = np.random.lognormal(mean=8.2, sigma=1.3, size=n_fraud)
        fraud_amounts = np.round(np.clip(fraud_amounts, 500.0, 500000.0), 2)

        partial_drain_mask = np.random.rand(n_fraud) < 0.45
        fraud_oldbalance_orig = fraud_amounts.copy()
        fraud_oldbalance_orig[partial_drain_mask] = np.round(fraud_amounts[partial_drain_mask] * np.random.uniform(1.05, 3.0, np.sum(partial_drain_mask)), 2)
        fraud_newbalance_orig = np.zeros(n_fraud)
        fraud_newbalance_orig[partial_drain_mask] = np.round(fraud_oldbalance_orig[partial_drain_mask] - fraud_amounts[partial_drain_mask], 2)

        fraud_oldbalance_dest = np.random.choice([0.0, 5000.0, 25000.0, 80000.0], size=n_fraud, p=[0.50, 0.25, 0.15, 0.10])
        fraud_newbalance_dest = np.zeros(n_fraud)
        dest_keep = np.random.rand(n_fraud) > 0.45
        fraud_newbalance_dest[dest_keep] = fraud_oldbalance_dest[dest_keep] + fraud_amounts[dest_keep]

        fraud_steps = np.random.randint(1, 721, size=n_fraud)
        
        mule_accounts = [f"MULE_{i:03d}" for i in range(1, 40)]
        fraud_dests = []
        for _ in range(n_fraud):
            if np.random.rand() > 0.25:
                fraud_dests.append(np.random.choice(mule_accounts))
            else:
                fraud_dests.append(f"C{np.random.randint(10000, 99999)}")
        
        fraud_origs = [f"VICTIM_{np.random.randint(1000, 9999)}" for _ in range(n_fraud)]

        fraud_df = pd.DataFrame({
            'step': fraud_steps,
            'type': fraud_types,
            'amount': fraud_amounts,
            'nameOrig': fraud_origs,
            'oldbalanceOrg': fraud_oldbalance_orig,
            'newbalanceOrig': fraud_newbalance_orig,
            'nameDest': fraud_dests,
            'oldbalanceDest': fraud_oldbalance_dest,
            'newbalanceDest': fraud_newbalance_dest,
            'isFraud': 1
        })

        df = pd.concat([legit_df, fraud_df], ignore_index=True)
        return df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['errorBalanceOrig'] = (df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']).astype(np.float32)
        df['errorBalanceDest'] = (df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']).astype(np.float32)

        safe_old_balance = np.where(df['oldbalanceOrg'] <= 0, 1.0, df['oldbalanceOrg'])
        df['amount_to_oldbalance_ratio'] = (df['amount'] / safe_old_balance).clip(upper=100.0).astype(np.float32)

        df['is_zero_balance_drain'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(np.float32)

        hour_of_day = (df['step'] % 24).astype(np.float32)
        df['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24.0).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24.0).astype(np.float32)

        orig_counts = df.groupby('nameOrig')['step'].transform('count')
        dest_counts = df.groupby('nameDest')['step'].transform('count')
        df['orig_velocity_1h'] = np.log1p(orig_counts).astype(np.float32)
        df['dest_velocity_1h'] = np.log1p(dest_counts).astype(np.float32)

        return df

    def fit_transform_split(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        df_featured = self.engineer_features(df)
        y = df_featured['isFraud'].values
        X_raw = df_featured[self.numeric_features + self.categorical_features]

        X_train_raw, X_test_raw, y_train, y_test, df_train, df_test = train_test_split(
            X_raw, y, df_featured,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
            ]
        )

        X_train_trans = self.preprocessor.fit_transform(X_train_raw)
        X_test_trans = self.preprocessor.transform(X_test_raw)

        cat_encoder = self.preprocessor.named_transformers_['cat']
        encoded_cats = [f"type_{c}" for c in cat_encoder.categories_[0]]
        self.feature_names = self.numeric_features + encoded_cats

        return X_train_trans, X_test_trans, y_train, y_test, df_train, df_test

    def transform_single(self, transaction: Dict[str, Any]) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted.")

        single_df = pd.DataFrame([transaction])
        if 'step' not in single_df.columns:
            single_df['step'] = 12
        if 'nameOrig' not in single_df.columns:
            single_df['nameOrig'] = 'C_UNKNOWN'
        if 'nameDest' not in single_df.columns:
            single_df['nameDest'] = 'M_UNKNOWN'
            
        featured = self.engineer_features(single_df)
        X_raw = featured[self.numeric_features + self.categorical_features]
        return self.preprocessor.transform(X_raw)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.preprocessor = data['preprocessor']
        self.feature_names = data['feature_names']
        self.numeric_features = data['numeric_features']
        self.categorical_features = data['categorical_features']
