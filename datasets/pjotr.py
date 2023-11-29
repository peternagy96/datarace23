import pandas as pd
pd.options.mode.chained_assignment = None
from .base import Dataset

class PjotrDataset(Dataset):
    def do_feature_engineering(self, df: pd.DataFrame):
        df['num_contracts'] = df.groupby('BORROWER_ID')['CONTRACT_ID'].transform('count')
        df['num_borrowers'] = df.groupby('CONTRACT_ID')['BORROWER_ID'].transform('count')
        
        df["CREDIT_INTERMEDIARY_AND_REFINANCED"] = (
            df["CONTRACT_CREDIT_INTERMEDIARY"].astype(str)
            + "_"
            + df["CONTRACT_REFINANCED"].astype(str)
        )
        
        # combine CONTRACT_CREDIT_INTERMEDIARY and CONTRACT_CREDIT_REFINANCED, corr: 7% & 7% -> -8%
        df["CREDIT_INTERMEDIARY_AND_REFINANCED"] = (
            df["CONTRACT_CREDIT_INTERMEDIARY"].astype(str)
            + "_"
            + df["CONTRACT_REFINANCED"].astype(str)
        )

        # new feature to indicate num of contracts per borrower until that date
        # so for a borrower, this feature should be 0 for his first contract, 1 for his second contract, etc.
        df["CONTRACT_COUNT"] = df.sort_values(["BORROWER_ID", "CONTRACT_DATE_OF_LOAN_AGREEMENT"]).BORROWER_ID.groupby(
            df["BORROWER_ID"]
        ).cumcount()
        df.loc[df["BORROWER_ID"]  == 'xNullx', "CONTRACT_COUNT"] = 0


        # new feature to indicate cumulative credit amount per borrower until that date
        df["CONTRACT_CUMULATIVE_CREDIT_AMOUNT"] = df.sort_values(["BORROWER_ID", "CONTRACT_DATE_OF_LOAN_AGREEMENT"]).CONTRACT_LOAN_AMOUNT.groupby(
            df["BORROWER_ID"]
        ).cumsum()
        df.loc[df["BORROWER_ID"]  == 'xNullx', "CONTRACT_CUMULATIVE_CREDIT_AMOUNT"] = 0

        # new feature to indicate number of unique types of credit per borrower until that date
        df["CONTRACT_LOAN_TYPE_UNIQUE_COUNT"] = df.sort_values(["BORROWER_ID", "CONTRACT_DATE_OF_LOAN_AGREEMENT"])\
            .drop_duplicates(subset=["BORROWER_ID", "CONTRACT_LOAN_TYPE"]).BORROWER_ID.groupby(
            df["BORROWER_ID"]
        ).cumcount()
        df.loc[df["BORROWER_ID"]  == 'xNullx', "CONTRACT_LOAN_TYPE_UNIQUE_COUNT"] = 0

        for col in (["CONTRACT_RISK_WEIGHTED_ASSETS",
                      "CONTRACT_INSTALMENT_AMOUNT",
                      "CONTRACT_INSTALMENT_AMOUNT_2",
                      "CONTRACT_LGD"] + self.log_transform_features):

            df[f'sum_{col}'] = df.groupby('BORROWER_ID')[col].transform('sum')
            
        return df
