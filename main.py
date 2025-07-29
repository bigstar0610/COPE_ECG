"""
Simplified HRV-based classification model for cardiac event prediction.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from bayes_opt import BayesianOptimization
import shap

SEED = 42

LGBM_BOUNDS = {
    'num_leaves': (16, 32),
    'lambda_l1': (0.7, 0.9),
    'lambda_l2': (0.9, 1.0),
    'feature_fraction': (0.6, 0.7),
    'bagging_fraction': (0.6, 0.9),
    'min_child_samples': (6, 10),
    'min_child_weight': (10, 40)
}

RF_BOUNDS = {
    'n_estimators': (50, 300),
    'max_depth': (3, 20),
    'max_features': (0.1, 1.0)
}

LR_BOUNDS = {
    'C': (1e-3, 10)
}

LGBM_FIXED = {
    'objective': 'binary',
    'learning_rate': 0.005,
    'bagging_freq': 1,
    'force_row_wise': True,
    'max_depth': 5,
    'verbose': -1,
    'random_state': SEED,
    'n_jobs': -1
}

def load_and_prepare_data(data_path, hours):
    """Load and prepare data for the specified time interval."""
    df = pd.read_csv(data_path)
    
    target_segment = hours * 60  
    df_filtered = df[(df['Time_segment'] > target_segment - 30) & 
                     (df['Time_segment'] <= target_segment)].copy()
    
    feature_cols = [col for col in df.columns if col.startswith('HRV_')]
    filled_data = pd.DataFrame()
    
    for case_id, case_group in df_filtered.groupby('Case_ID'):
        case_data = case_group.copy()
        for (time_seg, label), group in case_group.groupby(['Time_segment', 'Label']):
            medians = group[feature_cols].median(skipna=True)
            mask = (case_data['Time_segment'] == time_seg) & (case_data['Label'] == label)
            for col in feature_cols:
                case_data.loc[mask & case_data[col].isna(), col] = medians[col]
        filled_data = pd.concat([filled_data, case_data.dropna(subset=feature_cols)])
    
    X = filled_data[feature_cols]
    y = filled_data['Label']
    
    case_ids = filled_data['Case_ID'].unique()
    case_labels = {cid: filled_data[filled_data['Case_ID'] == cid]['Label'].mode()[0] 
                   for cid in case_ids}
    case_df = pd.DataFrame(list(case_labels.items()), columns=['Case_ID', 'Label'])
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_cases, test_cases = next(splitter.split(case_df[['Case_ID']], case_df['Label']))
    
    train_case_ids = case_df.iloc[train_cases]['Case_ID'].values
    test_case_ids = case_df.iloc[test_cases]['Case_ID'].values
    
    train_mask = filled_data['Case_ID'].isin(train_case_ids)
    test_mask = filled_data['Case_ID'].isin(test_case_ids)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                  columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                 columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, y_train, X_test_scaled, y_test

def feature_selection_boruta(X_train, y_train, n_trials=20):
    """Simplified BorutaShap feature selection."""
    features = list(X_train.columns)
    accepted = []
    history = {f: [] for f in features}
    
    for trial in range(n_trials):
        X_shadow = X_train.copy()
        for col in features:
            X_shadow[f"shadow_{col}"] = np.random.permutation(X_train[col].values)
        
        model = lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbosity=-1)
        model.fit(X_shadow, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shadow)
        if isinstance(shap_values, list):
            importance = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
        
        feature_imp = importance[:len(features)]
        shadow_imp = importance[len(features):]
        max_shadow = np.max(shadow_imp)
        
        for i, feature in enumerate(features):
            history[feature].append(feature_imp[i])
            if (len(history[feature]) >= 8 and 
                np.mean([h > max_shadow for h in history[feature][-8:]]) >= 0.7):
                if feature not in accepted:
                    accepted.append(feature)
    
    return accepted[:15]

def optimize_hyperparameters(X_train, y_train, model_name):
    """Bayesian optimization for hyperparameters."""
    if model_name == 'lgbm':
        bounds = LGBM_BOUNDS
        def eval_fn(num_leaves, lambda_l1, lambda_l2, feature_fraction, 
                    bagging_fraction, min_child_samples, min_child_weight):
            params = {
                'num_leaves': int(round(num_leaves)),
                'lambda_l1': lambda_l1,
                'lambda_l2': lambda_l2,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'min_child_samples': int(round(min_child_samples)),
                'min_child_weight': min_child_weight,
                **LGBM_FIXED
            }
            model = lgb.LGBMClassifier(**params)
            return cv_score(model, X_train, y_train)
    
    elif model_name == 'rf':
        bounds = RF_BOUNDS
        def eval_fn(n_estimators, max_depth, max_features):
            params = {
                'n_estimators': int(round(n_estimators)),
                'max_depth': int(round(max_depth)),
                'max_features': max_features,
                'random_state': SEED,
                'n_jobs': -1
            }
            model = RandomForestClassifier(**params)
            return cv_score(model, X_train, y_train)
    
    else:  # lr
        bounds = LR_BOUNDS
        def eval_fn(C):
            params = {'C': C, 'solver': 'liblinear', 'random_state': SEED}
            model = LogisticRegression(**params)
            return cv_score(model, X_train, y_train)
    
    optimizer = BayesianOptimization(f=eval_fn, pbounds=bounds, random_state=SEED)
    optimizer.maximize(init_points=3, n_iter=15)
    best_params = optimizer.max['params']
    
    if model_name == 'lgbm':
        best_params['num_leaves'] = int(round(best_params['num_leaves']))
        best_params['min_child_samples'] = int(round(best_params['min_child_samples']))
        best_params.update(LGBM_FIXED)
        
    elif model_name == 'rf':
        best_params['n_estimators'] = int(round(best_params['n_estimators']))
        best_params['max_depth'] = int(round(best_params['max_depth']))
        best_params.update({'random_state': SEED, 'n_jobs': -1})
        
    else:
        best_params.update({'solver': 'liblinear', 'random_state': SEED})
    
    return best_params

def cv_score(model, X_train, y_train):
    """Cross-validation scoring."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, y_pred))
    
    return np.mean(scores)

def train_and_evaluate(X_train, y_train, X_test, y_test, model_name, params):
    """Train final model and evaluate performance."""
    
    if model_name == 'lgbm':
        model = lgb.LGBMClassifier(**params)
        
    elif model_name == 'rf':
        model = RandomForestClassifier(**params)
        
    else:  # lr
        model = LogisticRegression(**params)
    
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'model': model_name,
        'auroc': auroc,
        'auprc': auprc,
        'threshold': optimal_threshold,
        'y_pred_proba': y_pred_proba.tolist()
    }, model

def main():
    """Main pipeline execution."""
    data_path = 'data.csv'
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    time_intervals = [2.0, 3.0, 6.0, 8.0, 10.0, 11.5]
    all_results = []
    
    all_features = set()
    for hours in time_intervals[:3]:
        X_train, y_train, _, _ = load_and_prepare_data(data_path, hours)
        selected = feature_selection_boruta(X_train, y_train, n_trials=15)
        all_features.update(selected)
    final_features = list(all_features)[:15]
    
    for hours in time_intervals:
        X_train, y_train, X_test, y_test = load_and_prepare_data(data_path, hours)
        X_train_selected = X_train[final_features]
        X_test_selected = X_test[final_features]
        
        best_params = optimize_hyperparameters(X_train_selected, y_train)    
        results, model = train_and_evaluate(X_train_selected, y_train, 
                                          X_test_selected, y_test, best_params)
        results['hours'] = hours
        all_results.append(results)
        
        with open(output_dir / f"results_{hours}hr.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

if __name__ == '__main__':
    main()
