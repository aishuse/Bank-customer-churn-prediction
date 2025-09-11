import os
import pandas as pd
import joblib
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from src.logger import logging
from src.exception import CustomException

def train_models(train_path: str, test_path: str, target_col: str = "Attrition_Flag"):
    try:
        logging.info("Loading prepared data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # ==========================
        # 1️⃣ RFECV Feature Selection
        # ==========================
        logging.info("Starting RFECV feature selection...")
        rfecv = RFECV(
            estimator=lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=-1),
            step=1,
            cv=cv,
            scoring='recall',
            min_features_to_select=5,
            n_jobs=-1
        )
        rfecv.fit(X_train, y_train)
        selected_features = X_train.columns[rfecv.support_].tolist()
        X_train_sel = rfecv.transform(X_train)
        X_test_sel = rfecv.transform(X_test)
        logging.info(f"Selected features: {selected_features}")

        # ==========================
        # 2️⃣ Train/Validation split for early stopping
        # ==========================
        X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
            X_train_sel, y_train, test_size=0.1, stratify=y_train, random_state=42
        )

        # ==========================
        # 3️⃣ Hyperparameter tuning
        # ==========================
        lgb_param_grid = {
            'num_leaves': [15, 31, 50],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [500, 1000, 2000],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.0, 0.5, 1.0],
            'reg_lambda': [0.0, 1.0, 3.0]
        }

        xgb_param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [500, 1000, 2000],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.0, 0.5, 1.0],
            'reg_lambda': [0.0, 1.0, 3.0]
        }

        # LightGBM RandomizedSearchCV
        logging.info("Tuning LightGBM...")
        lgb_search = RandomizedSearchCV(
            estimator=lgb.LGBMClassifier(random_state=42),
            param_distributions=lgb_param_grid,
            n_iter=20,
            scoring='roc_auc',
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        lgb_search.fit(X_train_sub, y_train_sub)

        # XGBoost RandomizedSearchCV
        logging.info("Tuning XGBoost...")
        xgb_search = RandomizedSearchCV(
            estimator=XGBClassifier(eval_metric='auc', random_state=42),
            param_distributions=xgb_param_grid,
            n_iter=20,
            scoring='roc_auc',
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        xgb_search.fit(X_train_sub, y_train_sub)

        # ==========================
        # 4️⃣ Final model training
        # ==========================
        best_model_name = "LightGBM" if lgb_search.best_score_ >= xgb_search.best_score_ else "XGBoost"

        if best_model_name == "LightGBM":
            best_params = lgb_search.best_params_
            best_params['n_estimators'] = 20000  # override for early stopping
            final_model = lgb.LGBMClassifier(**best_params, random_state=42)
            final_model.fit(
                X_train_sub, y_train_sub,
                eval_set=[(X_val_sub, y_val_sub)],
                eval_metric='auc',
                callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=100)]
            )
        else:
            best_params = xgb_search.best_params_
            final_model = XGBClassifier(
                **best_params,
                eval_metric='auc',
                n_estimators=2000,
                random_state=42
            )
            final_model.fit(
                X_train_sub, y_train_sub,
                eval_set=[(X_val_sub, y_val_sub)],
                early_stopping_rounds=100,
                verbose=100
            )

        logging.info(f"Selected best model: {best_model_name}")

        # ==========================
        # 5️⃣ Evaluate on Test Set
        # ==========================
        y_pred = final_model.predict(X_test_sel)
        y_proba = final_model.predict_proba(X_test_sel)[:, 1]

        logging.info("\n" + classification_report(y_test, y_pred))
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        logging.info(f"Final Test Metrics: {metrics}")

        # ==========================
        # 6️⃣ Save artifacts
        # ==========================
        artifacts_dir = os.path.join(os.getcwd(), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        joblib.dump(final_model, os.path.join(artifacts_dir, "best_model.pkl"))
        joblib.dump(selected_features, os.path.join(artifacts_dir, "selected_features.pkl"))
        logging.info(f"Saved best model and selected features in {artifacts_dir}/")

        return metrics

    except Exception as e:
        raise CustomException(error_detail=e, error_message=str(e))

if __name__ == "__main__":
    train_path = "data/prepared/train_prepared.csv"
    test_path = "data/prepared/test_prepared.csv"
    train_models(train_path, test_path)
