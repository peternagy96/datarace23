from .base import Dataset

class PjotrDataset(Dataset):
    def basic_transformations(self):
        """Implement all basic transformations here."""
        self.convert_to_datetime()
        self.create_label()

    def do_feature_engineering(self):
        self.train_df['num_contracts'] = self.train_df.groupby('BORROWER_ID')['CONTRACT_ID'].transform('count')
        self.train_df['num_borrowers'] = self.train_df.groupby('CONTRACT_ID')['BORROWER_ID'].transform('count')

        for col in (["CONTRACT_RISK_WEIGHTED_ASSETS",
                      "CONTRACT_INSTALMENT_AMOUNT",
                      "CONTRACT_INSTALMENT_AMOUNT_2",
                      "CONTRACT_LGD"] + self.log_transform_features):

            self.train_df[f'sum_{col}'] = self.train_df.groupby('BORROWER_ID')[col].transform('sum')

        self.feature_engineered_cols = [col for col in self.train_df.columns if col.startswith("sum") or col.startswith("num")]