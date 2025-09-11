import numpy as np
import pandas as pd

import os

# fetch the data from data/raw
train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')

def feature_engineering(df_eng: pd.DataFrame) -> pd.DataFrame:
    # Create new features
    df_eng['High_Utilization'] = (df_eng['Avg_Utilization_Ratio'] > 0.7).astype(int)
    df_eng['Activity_Decline'] = ((df_eng['Total_Ct_Chng_Q4_Q1'] < 0.7) |
                                (df_eng['Total_Amt_Chng_Q4_Q1'] < 0.7)).astype(int)
    df_eng['Low_Engagement'] = ((df_eng['Months_Inactive_12_mon'] > 2) &
                                (df_eng['Contacts_Count_12_mon'] < 3)).astype(int)
    df_eng['Avg_Transaction_Amount'] = df_eng['Total_Trans_Amt'] / df_eng['Total_Trans_Ct']
    df_eng['Credit_Dependency'] = df_eng['Total_Revolving_Bal'] / df_eng['Credit_Limit']
    df_eng['High_Value_Customer'] = (df_eng['Total_Trans_Amt'] >
                                    df_eng['Total_Trans_Amt'].quantile(0.75)).astype(int)
    df_eng['Multi_Product_User'] = (df_eng['Total_Relationship_Count'] > 3).astype(int)
    return df_eng

def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded



train_processed_data = one_hot_encoding(feature_engineering(train_data))
test_processed_data = one_hot_encoding(feature_engineering(test_data))

# store the data inside data/processed
data_path = os.path.join("data","processed")

os.makedirs(data_path, exist_ok=True)

train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"), index=False)
test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"), index=False)