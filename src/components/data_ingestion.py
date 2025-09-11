import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
from src.exception import CustomException
from src.logger import logging


def load_data(data_url: str) -> pd.DataFrame:
    
    logging.info("Entered the data ingestion method or component")
    try:
        df = pd.read_csv(data_url)
        logging.info('Read the dataset as dataframe')

        return df
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse the CSV file from {data_url}.")
        print(e)
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading the data.")
        print(e)
        raise



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.iloc[:,:-2] #drop the last two columns
        df = df.drop(columns=['CLIENTNUM'])
        df['Attrition_Flag'] = df['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
        df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
        return df
    except KeyError as e:
        print(f"Error: Missing column {e} in the dataframe.")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred during preprocessing.")
        print(e)
        raise




def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logging.info("Ingestion of the data is completed")
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise


def main():
    try:
        df = load_data(data_url='data/BankChurners.csv')
        final_df = preprocess_data(df)
        logging.info("Train test split initiated")
        train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)
        save_data(train_data, test_data, data_path='data')
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to complete the data ingestion process.")

if __name__ == '__main__':
    main()