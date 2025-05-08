import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from scipy.stats import ttest_ind, chi2, kendalltau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from bayes_opt import BayesianOptimization

SEED = 42

def analyze_hrv():
    df = pd.read_csv('data.csv')
    
    results = []
    
    hrvs = [col for col in df.columns if col.startswith('HRV_')]

    for param in hrvs:
        cases = df[df['Label'] == 1].copy()
        controls = df[df['Label'] == 0].copy()
        
        grouped = cases.groupby('Time')[param]
        means = grouped.mean()
        stds = grouped.std()
        counts = grouped.count()
        
        valid = ~means.isna() & (means != 0)
        times = means[valid].index.tolist()
        
        control_means = controls.groupby('Time')[param].mean()
        
        tstats = []
        pvals = []
        
        for t in times:
            case_data = cases[cases['Time'] == t][param].dropna().values
            control_data = controls[controls['Time'] == t][param].dropna().values
            
            if len(case_data) > 0 and len(control_data) > 0:
                tstat, pval = ttest_ind(case_data, control_data, equal_var=False)
                tstats.append(tstat)
                pvals.append(pval)
            else:
                tstats.append(np.nan)
                pvals.append(np.nan)
        
        valid_pvals = [p for p in pvals if not np.isnan(p)]
        if valid_pvals:
            fisher = -2 * np.sum(np.log(valid_pvals))
            df_fisher = 2 * len(valid_pvals)
            combined_p = 1 - chi2.cdf(fisher, df_fisher)
        else:
            combined_p = np.nan
            fisher = np.nan
        
        valid_cases = cases.dropna(subset=[param])
        tau, tau_p = kendalltau(valid_cases['Time'], valid_cases[param])
        
        result = {
            'Param': param,
            'Fisher': fisher,
            'CombinedP': combined_p,
            'Significant': combined_p < 0.001 if not np.isnan(combined_p) else False,
            'Tau': tau,
            'TauP': tau_p
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    return results_df

def train_model():
    data = pd.read_csv('data.csv', index_col=0)
    
    y = data['label']
    train_idx = data[data['test']==0].index
    test_idx = data[data['test']==1].index
    
    X = data.drop(columns=['stayid', 'label', 'time', 'test'])
    Xtrain, Xtest = X.loc[train_idx], X.loc[test_idx]
    ytrain, ytest = y.loc[train_idx], y.loc[test_idx]
    
    dtrain = lgb.Dataset(Xtrain, ytrain)
    dtest = lgb.Dataset(Xtest, ytest)
    
    model = lgb.LGBMClassifier(random_state=SEED, n_jobs=-1)
    model.fit(Xtrain, ytrain)
    
    explainer = shap.TreeExplainer(model)
    shaps = explainer.shap_values(Xtrain)
    
    if isinstance(shaps, list):
        importance = np.abs(np.array(shaps)).mean(axis=0).mean(axis=0)
    else:
        importance = np.abs(shaps).mean(axis=0)
    
    features = pd.DataFrame({
        'Feature': Xtrain.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    topfeats = features['Feature'].head(15).tolist()
    
    bounds = {
        'num_leaves': (16, 32),
        'lambda_l1': (0.7, 0.9),
        'lambda_l2': (0.9, 1.0),
        'feature_fraction': (0.6, 0.7),
        'bagging_fraction': (0.6, 0.9),
        'min_child_samples': (6, 10),
        'min_child_weight': (10, 40)
    }
    
    params = {
        'objective': 'binary',
        'learning_rate': 0.005,
        'bagging_freq': 1,
        'force_row_wise': True,
        'max_depth': 5,
        'verbose': -1,
        'random_state': SEED,
        'n_jobs': -1
    }
    
    def auprc(preds, dtrain):
        labels = dtrain.get_label()
        return 'auprc', average_precision_score(labels, preds), True
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    Xvals, yvals = Xtrain.values.astype(float), ytrain.values.flatten().astype(bool)
    
    best_params = []
    best_scores = []
    
    for idx, (train_i, valid_i) in enumerate(folds.split(Xvals, yvals)):
        Xfold, yfold = Xvals[train_i], yvals[train_i]
        Xval, yval = Xvals[valid_i], yvals[valid_i]
        
        train_ds = lgb.Dataset(Xfold, yfold)
        valid_ds = lgb.Dataset(Xval, yval)
        
        def eval_fn(num_leaves, lambda_l1, lambda_l2, feature_fraction, 
                  bagging_fraction, min_child_samples, min_child_weight):
            p = {
                'num_leaves': int(round(num_leaves)),
                'lambda_l1': lambda_l1,
                'lambda_l2': lambda_l2,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'min_child_samples': int(round(min_child_samples)),
                'min_child_weight': min_child_weight,
                'feature_pre_filter': False,
                **params
            }
            
            m = lgb.train(
                params=p,
                train_set=train_ds,
                num_boost_round=1000,
                valid_sets=[valid_ds],
                feval=auprc,
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            pred = m.predict(Xval)
            score = average_precision_score(yval, pred)
            return score
        
        opt = BayesianOptimization(
            f=eval_fn,
            pbounds=bounds,
            random_state=SEED
        )
        
        opt.maximize(init_points=3, n_iter=20)
        
        best_p = opt.max['params']
        best_s = opt.max['target']
        best_params.append(best_p)
        best_scores.append(best_s)
    
    best_fold = np.argmax(best_scores)
    best_p = best_params[best_fold]
    
    best_p['num_leaves'] = int(round(best_p['num_leaves']))
    best_p['min_child_samples'] = int(round(best_p['min_child_samples']))
    best_p.update(params)
    
    iters = []
    
    for idx, (train_i, valid_i) in enumerate(folds.split(Xvals, yvals)):
        Xfold, yfold = Xvals[train_i], yvals[train_i]
        Xval, yval = Xvals[valid_i], yvals[valid_i]
        
        train_ds = lgb.Dataset(Xfold, yfold)
        valid_ds = lgb.Dataset(Xval, yval)
        
        m = lgb.train(
            params=best_p,
            train_set=train_ds,
            num_boost_round=2000,
            valid_sets=[valid_ds],
            feval=auprc,
            early_stopping_rounds=100
        )
        
        iters.append(m.best_iteration)
    
    final_iter = int(np.mean(iters))
    
    final = lgb.train(
        params=best_p,
        train_set=dtrain,
        num_boost_round=final_iter
    )
    
    probs = final.predict(Xtest, num_iteration=final_iter)
    
    auroc = roc_auc_score(ytest, probs)
    auprc = average_precision_score(ytest, probs)
    
    results = {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'features': topfeats,
        'iters': final_iter
    }
    
    return results

def main():
    stats = analyze_hrv()
    
    pred = train_model()
    
    print("T-test Statistical Results for HRV Parameters:")
    for param, row in stats.iterrows():
        if row['CombinedP'] < 0.05:
            print(f"Parameter: {row['Param']}, p-value: {row['CombinedP']:.6f}")
    
    print("\nPrediction Model Performance:")
    print(f"AUROC: {pred['auroc']:.4f}")
    print(f"AUPRC: {pred['auprc']:.4f}")

if __name__ == '__main__':
    main()
