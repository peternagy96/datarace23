from .base import Dataset


class PetyaDataset(Dataset):
    def do_feature_engineering(self):
        self.train_df["num_borrowers"] = self.train_df.groupby("CONTRACT_ID")[
            "BORROWER_ID"
        ].transform("count")

        # combine CONTRACT_CREDIT_INTERMEDIARY and CONTRACT_CREDIT_REFINANCED, corr: 7% & 7% -> -8%
        self.train_df["CREDIT_INTERMEDIARY_AND_REFINANCED"] = (
            self.train_df["CONTRACT_CREDIT_INTERMEDIARY"].astype(str)
            + "_"
            + self.train_df["CONTRACT_CREDIT_REFINANCED"].astype(str)
        )

        # new feature to indicate num of contracts per borrower until that date
        # so for a borrower, this feature should be 0 for his first contract, 1 for his second contract, etc.
        self.train_df.sort_values(["BORROWER_ID", "CONTRACT_DATE_OF_LOAN_AGREEMENT"]).BORROWER_ID.groupby(
            self.train_df["BORROWER_ID"]
        ).cumcount()


        # ToDo: new feature to indicate cumulative credit amount per borrower until that date
        # ToDo: new feature to indicate number of types of credit per borrower until that date
        # ToDo: new feature to indicate sum of instalments per borrower until that date

        for col in [
            "CONTRACT_RISK_WEIGHTED_ASSETS",
            "CONTRACT_INSTALMENT_AMOUNT",
            "CONTRACT_INSTALMENT_AMOUNT_2",
            "CONTRACT_LGD",
        ] + self.log_transform_features:
            self.train_df[f"sum_{col}"] = self.train_df.groupby("BORROWER_ID")[
                col
            ].transform("sum")

        self.feature_engineered_cols = [
            col
            for col in self.train_df.columns
            if col.startswith("sum") or col.startswith("num")
        ]
