import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from src.logger import logging
from src.exception import CustomException

def prepare_data(processed_train_path: str, processed_test_path: str, target_col: str = "Attrition_Flag"):
    try:
        logging.info("Loading processed data...")
        train_df = pd.read_csv(processed_train_path)
        test_df = pd.read_csv(processed_test_path)

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        logging.info("Applying SMOTE to balance training data")
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        # Combine X and y to save as CSV
        train_sm_df = pd.DataFrame(X_train_sm, columns=X_train.columns)
        train_sm_df[target_col] = y_train_sm
        test_df[target_col] = y_test

        # Save CSVs
        os.makedirs("data/prepared", exist_ok=True)
        train_sm_df.to_csv("data/prepared/train_prepared.csv", index=False)
        test_df.to_csv("data/prepared/test_prepared.csv", index=False)

        logging.info("SMOTE and CSV saving done.")
        return train_sm_df, test_df

    except Exception as e:
        raise CustomException(e)

if __name__ == "__main__":
    train_path = "data/processed/train_processed.csv"
    test_path = "data/processed/test_processed.csv"
    prepare_data(train_path, test_path)
