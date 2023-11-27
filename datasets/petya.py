import pandas as pd
pd.options.mode.chained_assignment = None

from .base import Dataset


class PetyaDataset(Dataset):
    def do_feature_engineering(self, df: pd.DataFrame):
        df["NUM_BORROWERS"] = df.groupby("CONTRACT_ID")[
            "BORROWER_ID"
        ].transform("count")

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

        # new feature to indicate sum of instalments per borrower until that date
        df["CONTRACT_INSTALMENT_AMOUNT_SUM"] = df.sort_values(["BORROWER_ID", "CONTRACT_DATE_OF_LOAN_AGREEMENT"]).CONTRACT_INSTALMENT_AMOUNT.groupby(
            df["BORROWER_ID"]
        ).cumsum()
        df.loc[df["BORROWER_ID"]  == 'xNullx', "CONTRACT_INSTALMENT_AMOUNT_SUM"] = 0

        df["CONTRACT_INSTALMENT_AMOUNT_SUM_2"] = df.sort_values(["BORROWER_ID", "CONTRACT_DATE_OF_LOAN_AGREEMENT"]).CONTRACT_INSTALMENT_AMOUNT_2.groupby(
            df["BORROWER_ID"]
        ).cumsum()
        df.loc[df["BORROWER_ID"]  == 'xNullx', "CONTRACT_INSTALMENT_AMOUNT_SUM_2"] = 0

        return df
