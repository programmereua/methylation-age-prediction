import numpy as np
from pandas.io import feather_format
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.utils import resample
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from scipy.stats import spearmanr
from mrmr import mrmr_regression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, loguniform, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from mrmr import mrmr_classif
import optuna
from sklearn.model_selection import cross_val_score
import os
import pickle

#TASK 1.1
def load_csv(file_path): # load a CSV file

    print(f"\nLoading dataset from: {file_path}")
    df = pd.read_csv(file_path, index_col=0)
    print("The first rows of the dataset ", df.head())
    rows, cols = df.shape
    print("The dataset contains", rows, "samples and", cols, "columns.")
    return df

def split_development_data(df):

    print("Creating age bins.")
    age_bins = pd.qcut(df["age"], 5, duplicates="drop")  # divide ages into 5 groups the split is balanced

    # split 80% train, 20% validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=age_bins)

    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))

    return train_df, val_df


def load_data_preprocessing(development_path= "development_data.csv", evaluation_path= "evaluation_data.csv"):

    # Load development and evaluation datasets. Split development into train/validation.
    print("\nLoading development dataset")
    development_df = load_csv(development_path)

    print("\nLoading evaluation dataset")
    evaluation_df = load_csv(evaluation_path)

    #only for the development dataset
    train_df, val_df = split_development_data(development_df)
    print ("\n Splitting (80% TRAIN ,20% VAL) completed")
    return development_df, train_df, val_df, evaluation_df


#TASK 1.2

def check_missing_values(df):
    print("\nChecking missing values")

    missing = df.isna()
    total_missing = missing.sum().sum()

    print("Total missing values:", total_missing)
    missing_per_column = missing.sum()
    missing_sorted= missing_per_column.sort_values
    print("\nMissing values in column:",missing_sorted(ascending=False).head(10))

def get_feature_groups(df):
    metadata_cols = ["sex", "ethnicity"]
    cpg_cols = [col for col in df.columns if col.startswith("cg")]
    target_col = "age"
    return metadata_cols, cpg_cols, target_col

def get_feature_set(df, feature_set="all"):

    # metadata columns
    metadata_cols = ["sex", "ethnicity"]

    # CpG columns - all columns that start with "cg"
    cpg_cols = [col for col in df.columns if col.startswith("cg")]

    # return the right set of features
    if feature_set == "metadata":
        return metadata_cols

    elif feature_set == "cpg":
        return cpg_cols

    elif feature_set == "all":
        return metadata_cols + cpg_cols

    else:
        raise ValueError


def split_features_target(df, feature_cols, target_col= "age"):
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y


def build_preprocessor(cpg_cols, metadata_cols):

    transformers = []

    # for CpG columns: fill missing values with median, then scale
    if len(cpg_cols) > 0:
        cpg_steps = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ])
        transformers.append(("cpg", cpg_steps, cpg_cols))

    # for metadata columns: fill missing with most common value, then encode
    if len(metadata_cols) > 0:
        meta_steps = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("meta", meta_steps, metadata_cols))

    # combine both pipelines into one preprocessor
    preprocessor = ColumnTransformer(transformers)

    return preprocessor

def dataset_summary(df, name):
    print("\nDataset:", name)

    # number of samples
    n_samples = len(df)
    print("Number of samples:", n_samples)

    # age statistics
    age_mean = df["age"].mean()
    print("Age mean:", round(age_mean,2))

    age_std = df["age"].std()
    print("Age std:", round(age_std,2))

    age_min = df["age"].min()
    age_max = df["age"].max()
    print("Age range:", age_min, "-", age_max)

    # sex balance
    print("\nSex balance:")
    print(df["sex"].value_counts())

    # how many of each ethnicity
    print("\nEthnicity balance:")
    print(df["ethnicity"].value_counts())

    # how many CpG columns exist
    cpg_cols = [col for col in df.columns if col.startswith("cg")]
    print("\nNumber of CpG columns:", len(cpg_cols))

    # how many missing values in total
    total_missing = df.isna().sum().sum()
    print("Total missing values:", total_missing)


def plot_age_histogram(age):
    plt.hist(age, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Age distribution')
    plt.savefig('age_histogram_development.png', dpi=150, bbox_inches='tight')
    plt.show()
    
#TASK 1.3
def print_stats_table(train_df, val_df, evaluation_df):
    rows = []
    for name, df in [("Train", train_df), ("Validation", val_df), ("Evaluation", evaluation_df)]:
        rows.append({
            "Split"        : name,
            "N"            : len(df),
            "Age Mean±Std" : f"{df['age'].mean():.1f} ± {df['age'].std():.1f}",
            "Age Range"    : f"{df['age'].min():.0f} - {df['age'].max():.0f}",
            "Male"         : (df["sex"] == "M").sum(),
            "Female"       : (df["sex"] == "F").sum()
        })
    print(pd.DataFrame(rows).to_string(index=False))


def plot_age_by_split(train_df, val_df, evaluation_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, df) in zip(axes, [("Train", train_df), ("Validation", val_df), ("Evaluation", evaluation_df)]):
        ax.hist(df["age"], bins=20, color="steelblue", edgecolor="black")
        ax.set_title(name)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig('age_histogram_splits.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    
def exploratory_analysis(train_df, val_df, evaluation_df, development_df):
    print_stats_table(train_df, val_df, evaluation_df)
    plot_age_histogram(development_df["age"])
    plot_age_by_split(train_df, val_df, evaluation_df)
    print("\nMissing Values")
    check_missing_values(train_df)
    dataset_summary(train_df, "Training set")
    dataset_summary(val_df, "Validation set")
    dataset_summary(evaluation_df, "Evaluation set")

#TASK 2.1
def train_ols_model(X_train, y_train, X_val):

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return y_pred

def evaluate_model(y_val, y_pred):
    print("\nEvaluating model using bootstrap")

    np.random.seed(42)

  #creating empty list to store our metrics
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    pearson_scores = []

    for _ in range(1000):
        y_val_sample, y_pred_sample = resample(y_val, y_pred, replace=True)

        rmse = mean_squared_error(y_val_sample, y_pred_sample) ** 0.5
        mae = mean_absolute_error(y_val_sample, y_pred_sample)
        r2 = r2_score(y_val_sample, y_pred_sample)
        r, _ = pearsonr(y_val_sample, y_pred_sample)

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        pearson_scores.append(r)

    # metrics on full validation set
    full_rmse = mean_squared_error(y_val, y_pred) ** 0.5
    full_mae = mean_absolute_error(y_val, y_pred)
    full_r2 = r2_score(y_val, y_pred)
    full_r, _ = pearsonr(y_val, y_pred)

    # 95% confidence intervals
    rmse_ci = np.percentile(rmse_scores, [2.5, 97.5])
    mae_ci = np.percentile(mae_scores, [2.5, 97.5])
    r2_ci = np.percentile(r2_scores, [2.5, 97.5])
    pearson_ci = np.percentile(pearson_scores, [2.5, 97.5])

    print("\nValidation metrics with 95% CI:")
    print(f"RMSE: {full_rmse:.4f} (95% CI: {rmse_ci[0]:.4f} - {rmse_ci[1]:.4f})")
    print(f"MAE: {full_mae:.4f} (95% CI: {mae_ci[0]:.4f} - {mae_ci[1]:.4f})")
    print(f"R^2: {full_r2:.4f} (95% CI: {r2_ci[0]:.4f} - {r2_ci[1]:.4f})")
    print(f"Pearson r: {full_r:.4f} (95% CI: {pearson_ci[0]:.4f} - {pearson_ci[1]:.4f})")
    return {
    'rmse'        : full_rmse,
    'rmse_ci'     : rmse_ci,
    'rmse_scores' : rmse_scores,
    'mae'         : full_mae,
    'mae_ci'      : mae_ci,
    'r2'          : full_r2,
    'r2_ci'       : r2_ci,
    'r2_scores'   : r2_scores,
    'r'           : full_r,
    'pearson_ci'  : pearson_ci
    }

#TASK 2.2

#ElasticNet — L1+L2 regularised linear regression

def train_elastic_net_model(X_train, y_train, X_val):

    print("Elastic_net model")
    # Create an instance of the ElasticNet model
    elastic_net = ElasticNet() # default hyperparameters

    # Fit the model to the training data
    elastic_net.fit(X_train, y_train)
    print('Elastic Net model trained successfully.')

    # Make predictions on the test data
    y_pred = elastic_net.predict(X_val)
    print('Predictions made on the val data.')

    return y_pred


#SVR Support Vector Regression with RBF kernel

def train_SVR_model(X_train, y_train, X_val):
    print("SVR model with RBF kernel")

    SVR_model = SVR(kernel='rbf')
    SVR_model.fit(X_train,y_train)
    print('SVR_model succesfully done')
    y_pred = SVR_model.predict(X_val)
    print('Predictions made on the val data.')

    return y_pred

#Bayesian linear regression with automatic relevance determination

def train_BayesianRidge_model(X_train, y_train, X_val):
    print("Bayesian Ridge model")

    # create model (default hyperparameters)
    model = BayesianRidge()
    model.fit(X_train, y_train)
    print("Bayesian Ridge model traind successfully.")
    y_pred = model.predict(X_val)
    print("Predictions made on the val data.")

    return y_pred


def print_results_table(results_dict):
    rows = []
    for model_name, res in results_dict.items():
        rows.append({
            "Model"         : model_name,
            "RMSE (CI)"     : f"{res['rmse']:.3f} ({res['rmse_ci'][0]:.3f} - {res['rmse_ci'][1]:.3f})",
            "MAE (CI)"      : f"{res['mae']:.3f} ({res['mae_ci'][0]:.3f} - {res['mae_ci'][1]:.3f})",
            "R² (CI)"       : f"{res['r2']:.3f} ({res['r2_ci'][0]:.3f} - {res['r2_ci'][1]:.3f})",
            "Pearson r (CI)": f"{res['r']:.3f} ({res['pearson_ci'][0]:.3f} - {res['pearson_ci'][1]:.3f})"
        })
    df = pd.DataFrame(rows)
    print("\nModel Performance Summary:\n")
    print(df.to_string(index=False))


def plot_bootstrap_boxplots(results_dict):
    model_names = list(results_dict.keys())
    rmse_data   = [results_dict[m]["rmse_scores"] for m in model_names]
    r2_data     = [results_dict[m]["r2_scores"]   for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].boxplot(rmse_data)
    axes[0].set_xticks(range(1, len(model_names) + 1))
    axes[0].set_xticklabels(model_names)
    axes[0].set_title("Bootstrap RMSE Distribution")
    axes[0].set_ylabel("RMSE (years)")
    axes[0].set_xlabel("Model")

    axes[1].boxplot(r2_data)
    axes[1].set_xticks(range(1, len(model_names) + 1))
    axes[1].set_xticklabels(model_names)
    axes[1].set_title("Bootstrap R² Distribution")
    axes[1].set_ylabel("R²")
    axes[1].set_xlabel("Model")

    plt.tight_layout()
    plt.savefig('bootstrap_boxplots.png', dpi=150, bbox_inches='tight')
    plt.show()
    
#TASK 3.1
def stability_selection(train_df):

    cpg_cols = [col for col in train_df.columns if col.startswith("cg")]
    # counter
    counts = {}
    for col in cpg_cols:
        counts[col] = 0

    # FOR 50 times
    for i in range(50):

        # take a random 80% of the training data ,we dont use the prosessed one because it will have data leakage bedause the NAN values that where
        # replaced with the medians where  calculated from the whole X_train set.
        sub = train_df.sample(frac=0.8, replace=False, random_state=i)

        # fill missing values with median BEFORE calculating correlation
        imputer = SimpleImputer(strategy="median")
        sub_cpg = pd.DataFrame(imputer.fit_transform(sub[cpg_cols]), columns=cpg_cols)
        sub_age = sub["age"].values

        # calculate correlation of each CpG with age
        correlations = {}
        for col in cpg_cols:
            corr, _ = spearmanr(sub_cpg[col], sub_age)
            correlations[col] = abs(corr)
        # keep only the top 200 most correlated
        top200 = sorted(correlations, key=correlations.get, reverse=True)[:200]

        # add +1 to each winner
        for col in top200:
            counts[col] += 1

        print(f"Resample {i+1}/50 done")

    # convert to pandas Series
    counts = pd.Series(counts)

    # keep only features selected more than 25 times
    stable = {}
    for col in counts.index:
        if counts[col] > 25:
            stable[col] = counts[col]

    stable = pd.Series(stable)
    print(f"Stable features found: {len(stable)}")

    return stable, counts
     

def plot_frequency(counts):
    
    plt.figure(figsize=(8,5))
    plt.hist(counts.values, bins=20, color="steelblue", edgecolor="black")
    plt.axvline(25, color="red", linestyle="--", label="threshold = 25")
    plt.xlabel("Times selected (out of 50)")
    plt.ylabel("Number of CpGs")
    plt.title("Selection frequency distribution across CpG features")
    plt.legend()
    plt.savefig('stability_selection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
# TASK 3.2 Minimum Redundancy Maximum Relevance mRMR Feature Selection

#first lets find the best value of K

def choose_best_k(train_df, val_df):

    k_values = [50,75,90,100,120,150,200]  # we will try these K vals
        
    cpg_cols = [col for col in train_df.columns if col.startswith("cg")] #for all cg columns

    results = {} # store the RMSE for each K

    for K in k_values:
        
        print("Trying K=  ", K)

        # run mRMR and get the selected features
        selected = mrmr_regression(X=train_df[cpg_cols], y=train_df["age"], K=K)

        # preprocess the data using only the selected features
        preprocessor = build_preprocessor(selected, [])
        X_train_p = preprocessor.fit_transform(train_df[selected])
        X_val_p   = preprocessor.transform(val_df[selected])

        # step 3: train a simple model
        model = BayesianRidge()
        model.fit(X_train_p, train_df["age"])

        # predict on validation set
        y_pred = model.predict(X_val_p)

        # calculate RMSE - lower is better
        rmse = mean_squared_error(val_df["age"], y_pred) ** 0.5
        results[K] = rmse

        print(f"  K={K} gave RMSE = {rmse:.4f}")

    # find the best K = the one with lowest RMSE
    best_k = min(results, key=results.get)
    best_rmse = results[best_k]

    print(f"\nBest K = {best_k} with RMSE = {best_rmse:.4f}")
    print("(if two K values give similar RMSE, we prefer the smaller one because it is simpler)")

    plt.figure(figsize=(7, 4))

    # x axis- K values, y axis- RMSE for each K
    x = list(results.keys())
    y = list(results.values())

    plt.plot(x, y, marker="o", color="steelblue")

    plt.xlabel("K (number of features)")
    plt.ylabel("Validation RMSE")
    plt.title("mRMR - How many features do we need")
    plt.savefig('mrmr_k_selection.png', dpi=150, bbox_inches='tight')

    plt.show()

    return best_k
   
   
def run_mrmr(train_df, K):

    cpg_cols = [col for col in train_df.columns if col.startswith("cg")]   # get ALL CpG columns
    # take features and target
    X_train = train_df[cpg_cols]
    y_train = train_df["age"]

    selected_features = mrmr_regression(X=X_train, y=y_train, K=K) # run mRMR - finds K features that predict age but are different from each other
    
    # top 10
    print(f"mRMR selected {K} features")
    print("Top 10 features:")
    for i, feat in enumerate(selected_features[:10]):
      corr, _ = spearmanr(train_df[feat], train_df["age"])
      print(f"  {i+1}. {feat}  (Spearman r = {abs(corr):.4f})")

    return selected_features


def plot_overlap(features1, features2):
    
    #to compare them  
    set1 = set(features1)
    set2 = set(features2)
    
    # find what they share and what features not
    only_stability = set1 - set2
    overlap        = set1 & set2
    only_mrmr      = set2 - set1
    
    print('Stability only :', len(only_stability))
    print('Overlap        :', len(overlap))
    print('mRMR only      :', len(only_mrmr))
    
    # bar chart
    plt.figure(figsize=(6, 5))
    plt.bar(
        ['Stability only', 'Overlap', 'mRMR only'],
        [len(only_stability), len(overlap), len(only_mrmr)],
        color=['steelblue', 'green', 'orange'],
        edgecolor='black'
    )
    plt.title('Overlap between Stability and mRMR')
    plt.ylabel('Number of features')
    plt.xlabel('Feature group')
    plt.savefig('feature_overlap.png', dpi=150, bbox_inches='tight')
    plt.show()
    
#Task 3.3
def compare_feature_sets(train_df, val_df, stable_cpg_cols, mrmr_features):
    
    results = {}
    
    for name, features in [("Stability", stable_cpg_cols), ("mRMR", mrmr_features)]:
        
        # preprocess using only the selected features
        preprocessor = build_preprocessor(features, [])
        X_train_p = preprocessor.fit_transform(train_df[features])
        X_val_p   = preprocessor.transform(val_df[features])
        
        # get age values
        y_train = train_df["age"].values
        y_val   = val_df["age"].values
        
        # train 
        y_pred = train_BayesianRidge_model(X_train_p, y_train, X_val_p)
        
        #  metrics
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        r2   = r2_score(y_val, y_pred)
        
        # results
        results[name] = {
            "rmse"    : rmse,
            "r2"      : r2,
            "features": features,
            "n"       : len(features)  # how many features were selected
        }
        
        print(f"{name}: {len(features)} features, RMSE={rmse:.4f}, R2={r2:.4f}")
    
   
    print("\nComparison Table")
    print(f"{'Method':<12} {'N features':<12} {'RMSE':<10} {'R2'}")
    for name, res in results.items():
        print(f"{name:<12} {res['n']:<12} {res['rmse']:.4f}    {res['r2']:.4f}")
    
    # pick the lowest RMSE
    best_name = min(results, key=lambda x: results[x]["rmse"])
    print(f"\nBest method: {best_name}")
    print(f"RMSE: {results[best_name]['rmse']:.4f}")
    print(f"Number of features: {results[best_name]['n']}")
    print(f"This feature we will use for  Task 4")
    
    return results[best_name]["features"], best_name


    
# TASK 4.1
def tune_model(development_df, best_features):

    print("\nUsing the best selected feature set, we tune ElasticNet, SVR, and BayesianRidge with RandomizedSearchCV and 5-fold CV on the full development dataset.")

    # raw data only - no fit_transform here
    X_dev = development_df[best_features]
    y_dev = development_df["age"].values

    # ElasticNet pipeline
    elastic_pipe = Pipeline([
        ("preprocessor", build_preprocessor(best_features, [])),
        ("model", ElasticNet())
    ])

    elastic_search = RandomizedSearchCV(
        estimator=elastic_pipe,
        param_distributions={
            "model__alpha": loguniform(0.001, 10),
            "model__l1_ratio": uniform(0.1, 0.9)
        },
        n_iter=40,
        scoring="neg_root_mean_squared_error", # To minimize mean cross-validation Root Mean Squared Error (RMSE) in RandomizedSearchCV, set the scoring parameter to 'neg_root_mean_squared_error'
        cv=5,
        refit=True,
        random_state=42,
        n_jobs=-1
    )

    # SVR pipeline
    svr_pipe = Pipeline([
        ("preprocessor", build_preprocessor(best_features, [])),
        ("model", SVR())
    ])

    svr_search = RandomizedSearchCV(
        estimator=svr_pipe,
        param_distributions={
            "model__C": loguniform(0.1, 500),
            "model__epsilon": [0.01, 0.1, 0.5, 1.0],
            "model__kernel": ["rbf", "linear"]
        },
        n_iter=40,
        scoring="neg_root_mean_squared_error",
        cv=5,
        refit=True,
        random_state=42,
        n_jobs=-1
    )

    # BayesianRidge pipeline
    bayes_pipe = Pipeline([
        ("preprocessor", build_preprocessor(best_features, [])),
        ("model", BayesianRidge())
    ])

    bayes_search = RandomizedSearchCV(
        estimator=bayes_pipe,
        param_distributions={
            "model__alpha_1": loguniform(1e-7, 1e-3),
            "model__alpha_2": loguniform(1e-7, 1e-3),
            "model__lambda_1": loguniform(1e-7, 1e-3),
            "model__lambda_2": loguniform(1e-7, 1e-3)
        },
        n_iter=40,
        scoring="neg_root_mean_squared_error",
        cv=5,
        refit=True,
        random_state=42,
        n_jobs=-1
    )

    print("Tuning ElasticNet")
    elastic_search.fit(X_dev, y_dev)
    print("Best ElasticNet params:", elastic_search.best_params_)

    print("Tuning SVR")
    svr_search.fit(X_dev, y_dev)
    print("Best SVR params:", svr_search.best_params_)

    print("Tuning BayesianRidge")
    bayes_search.fit(X_dev, y_dev)
    print("Best BayesianRidge params:", bayes_search.best_params_)

    return elastic_search.best_estimator_, svr_search.best_estimator_, bayes_search.best_estimator_



# TASK 4.2
def evaluate_in_evaluation_data(model, evaluation_df, best_features):

    print("Now let's evaluate with  evaluation data")

    X_eval = evaluation_df[best_features]
    y_eval = evaluation_df["age"]

    # predictions from the already trained model
    y_eval_pred = model.predict(X_eval)

    np.random.seed(42) # for bootstrap as previously

    #for metrics
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    pearson_scores = []

    # for 1000 times
    for _ in range(1000):
        y_sample, y_pred_sample = resample(y_eval, y_eval_pred, replace=True)

        rmse = mean_squared_error(y_sample, y_pred_sample) ** 0.5
        mae = mean_absolute_error(y_sample, y_pred_sample)
        r2 = r2_score(y_sample, y_pred_sample)
        r, _ = pearsonr(y_sample, y_pred_sample)

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        pearson_scores.append(r)

    # metrics on full evaluation set
    full_rmse = mean_squared_error(y_eval, y_eval_pred) ** 0.5
    full_mae = mean_absolute_error(y_eval, y_eval_pred)
    full_r2 = r2_score(y_eval, y_eval_pred)
    full_r, _ = pearsonr(y_eval, y_eval_pred)

    # mean, std, and 95% CI
    rmse_mean = np.mean(rmse_scores)
    rmse_std = np.std(rmse_scores)
    rmse_ci = np.percentile(rmse_scores, [2.5, 97.5])

    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores)
    mae_ci = np.percentile(mae_scores, [2.5, 97.5])

    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    r2_ci = np.percentile(r2_scores, [2.5, 97.5])

    pearson_mean = np.mean(pearson_scores)
    pearson_std = np.std(pearson_scores)
    pearson_ci = np.percentile(pearson_scores, [2.5, 97.5])

    print("\nEvaluation metrics with bootstrap:")
    print(f"RMSE: mean={rmse_mean:.4f}, std={rmse_std:.4f}, 95% CI=({rmse_ci[0]:.4f}, {rmse_ci[1]:.4f})")
    print(f"MAE: mean={mae_mean:.4f}, std={mae_std:.4f}, 95% CI=({mae_ci[0]:.4f}, {mae_ci[1]:.4f})")
    print(f"R^2: mean={r2_mean:.4f}, std={r2_std:.4f}, 95% CI=({r2_ci[0]:.4f}, {r2_ci[1]:.4f})")
    print(f"Pearson r: mean={pearson_mean:.4f}, std={pearson_std:.4f}, 95% CI=({pearson_ci[0]:.4f}, {pearson_ci[1]:.4f})")

    return {
        "RMSE_mean": rmse_mean,
        "RMSE_std": rmse_std,
        "RMSE_CI": rmse_ci,
        "MAE_mean": mae_mean,
        "MAE_std": mae_std,
        "MAE_CI": mae_ci,
        "R2_mean": r2_mean,
        "R2_std": r2_std,
        "R2_CI": r2_ci,
        "Pearson_mean": pearson_mean,
        "Pearson_std": pearson_std,
        "Pearson_CI": pearson_ci
    }

#ΤΑSK 4.3

def save_best_model(elastic_results, svr_results, bayes_results, elastic_best, svr_best, bayes_best):
    
    # compare RMSE means and pick the best model
    results_summary = {
        'ElasticNet'   : elastic_results['RMSE_mean'],
        'SVR'          : svr_results['RMSE_mean'],
        'BayesianRidge': bayes_results['RMSE_mean']
    }
    
    # the best model is the one with the lowest RMSE
    best_model_name = min(results_summary, key=results_summary.get)
    print('Best model:', best_model_name)
    print('RMSE:', round(results_summary[best_model_name], 4))
    
    # get the actual model object
    all_models = {
        'ElasticNet'   : elastic_best,
        'SVR'          : svr_best,
        'BayesianRidge': bayes_best
    }
    best_model_obj = all_models[best_model_name]
    
    # create the models folder if it doesnt exist
    os.makedirs('../models', exist_ok=True)
    
    # save the best model to a file using pickle
    filename = '../models/best_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(best_model_obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Model saved to', filename)
    
    # load it back just to make sure it was saved correctly
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    print('Model loaded back successfully!')
    
    return best_model_name, best_model_obj


#BONUS A

# BONUS A - Optuna Hyperparameter Optimization
def optuna_tune_model(model_name, pipeline, X_train, y_train, n_trials=50, cv=5):
    
    # optuna will call this function many times
    # each time it tries different hyperparameters and we return the RMSE
    def objective(trial):
        
        # for each model we try different hyperparameters
        if model_name == "elasticnet":
            alpha    = trial.suggest_float("alpha",    0.001, 10,  log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.1,   1.0)
            pipeline.set_params(model__alpha=alpha, model__l1_ratio=l1_ratio)
            
        elif model_name == "svr":
            C       = trial.suggest_float("C", 0.1, 500, log=True)
            epsilon = trial.suggest_categorical("epsilon", [0.01, 0.1, 0.5, 1.0])
            kernel  = trial.suggest_categorical("kernel",  ["rbf", "linear"])
            pipeline.set_params(model__C=C, model__epsilon=epsilon, model__kernel=kernel)
            
        elif model_name == "bayesianridge":
            alpha_1  = trial.suggest_float("alpha_1",  1e-7, 1e-3, log=True)
            alpha_2  = trial.suggest_float("alpha_2",  1e-7, 1e-3, log=True)
            lambda_1 = trial.suggest_float("lambda_1", 1e-7, 1e-3, log=True)
            lambda_2 = trial.suggest_float("lambda_2", 1e-7, 1e-3, log=True)
            pipeline.set_params(
                model__alpha_1=alpha_1, model__alpha_2=alpha_2,
                model__lambda_1=lambda_1, model__lambda_2=lambda_2
            )
        
        # calculate the RMSE using cross validation
        # cross_val_score returns negative values so we flip the sign
        scores = cross_val_score(pipeline, X_train, y_train,
                                 scoring="neg_root_mean_squared_error", cv=cv)
        rmse = -scores.mean()
        return rmse
    
    # create the study and start searching
    print(f"Starting Optuna search for {model_name}...")
    print(f"We will try {n_trials} different combinations of hyperparameters")
    study = optuna.create_study(direction="minimize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # set the best hyperparameters and refit on full training data
    best = study.best_params
    if model_name == "elasticnet":
        pipeline.set_params(model__alpha=best["alpha"], model__l1_ratio=best["l1_ratio"])
    elif model_name == "svr":
        pipeline.set_params(model__C=best["C"], model__epsilon=best["epsilon"], model__kernel=best["kernel"])
    elif model_name == "bayesianridge":
        pipeline.set_params(
            model__alpha_1=best["alpha_1"], model__alpha_2=best["alpha_2"],
            model__lambda_1=best["lambda_1"], model__lambda_2=best["lambda_2"]
        )
    
    # train the final model with the best hyperparameters
    pipeline.fit(X_train, y_train)
    
    print(f"Done! Best RMSE found: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")
    
    return pipeline, study


def plot_optuna_history(study, model_name):
    
    # get the RMSE value for each trial
    trial_numbers = [t.number for t in study.trials]
    trial_rmse    = [t.value  for t in study.trials]
    
    plt.figure(figsize=(8, 4))
    
    # plot each trial as a dot
    plt.plot(trial_numbers, trial_rmse, 
             marker='o', color='steelblue', alpha=0.6, label='Each trial')
    
    # red line showing the best RMSE found
    plt.axhline(study.best_value, color='red', linestyle='--',
                label=f'Best RMSE = {study.best_value:.4f}')
    
    plt.xlabel('Trial number')
    plt.ylabel('RMSE')
    plt.title(f'Optuna search history - {model_name}')
    plt.legend()
    plt.savefig(f'optuna_history_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_optuna_comparison(elastic_results, svr_results, bayes_results,
                             elastic_study, svr_study, bayes_study):
    
    print("\nRandomSearch vs Optuna - Head to Head:")
    print(f"{'Model':<15} {'RandomSearch':<15} {'Optuna':<15} {'Delta':<10} {'Winner'}")
    print("-" * 65)
    
    models = [
        ("ElasticNet",    elastic_results, elastic_study),
        ("SVR",           svr_results,     svr_study),
        ("BayesianRidge", bayes_results,   bayes_study)
    ]
    
    for name, rs_res, opt_study in models:
        rs_rmse  = rs_res['RMSE_mean']
        opt_rmse = opt_study.best_value
        delta    = rs_rmse - opt_rmse
        
        # positive delta means optuna was better
        if opt_rmse < rs_rmse:
            winner = "Optuna"
        else:
            winner = "RandomSearch"
        
        print(f"{name:<15} {rs_rmse:<15.4f} {opt_rmse:<15.4f} {delta:<10.4f} {winner}")

        
#BONUS B
def create_sex_label(train_df, evaluation_df):

    # M = 1, F = 0
    train_df["sex_label"]      = [1 if label == "M" else 0 for label in train_df["sex"]]
    evaluation_df["sex_label"] = [1 if label == "M" else 0 for label in evaluation_df["sex"]]

    print("Sex labels created")
    return train_df, evaluation_df 


def select_sex_features(train_df, K):

    # get all CpG columns
    cpg_cols = [col for col in train_df.columns if col.startswith("cg")]

    # run mRMR but target is sex_label 
    print(f"Running mRMR for sex with K={K}")
    sex_features = mrmr_classif(X=train_df[cpg_cols], y=train_df["sex_label"], K=K)

    print(f"Selected {len(sex_features)} features for sex")

    return sex_features
    
def plot_sex_age_overlap(age_features, sex_features):

    age_set  = set(age_features)
    sex_set  = set(sex_features)

    overlap   = age_set & sex_set
    only_age  = age_set - sex_set
    only_sex  = sex_set - age_set

    print(f"Age only  : {len(only_age)}")
    print(f"Overlap   : {len(overlap)}")
    print(f"Sex only  : {len(only_sex)}")

    plt.figure(figsize=(6, 5))
    plt.bar(["Age only", "Overlap", "Sex only"],
            [len(only_age), len(overlap), len(only_sex)],
            color=["steelblue", "green", "orange"],
            edgecolor="black")
    plt.title("Age features vs Sex features - are they different?")
    plt.ylabel("Number of CpGs")
    plt.tight_layout()
    plt.savefig('sex_age_overlap.png', dpi=150, bbox_inches='tight')

    plt.show()
    
def train_classifiers(X_train, y_train):

    #Logistic Regression
    print("Training Logistic Regression")
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    print("Done!")

    # Gaussian Naive Bayes
    print("Training Gaussian Naive Bayes")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print("Done!")

    return lr, gnb
    

def evaluate_classifier(model, X_eval, y_eval):

    np.random.seed(42)

    # empty lists to store scores
    
    acc_scores = []
    f1_scores  = []
    mcc_scores = []
    roc_scores = []
    pr_scores  = []

    # bootstrap 1000 times
    for _ in range(1000):

        # take random sample with replacement
        X_sample, y_sample = resample(X_eval, y_eval, replace=True)

        # get predictions
        y_pred      = model.predict(X_sample)
        y_pred_prob = model.predict_proba(X_sample)[:, 1]

        # calculate metrics
        acc_scores.append(accuracy_score(y_sample, y_pred))
        f1_scores.append(f1_score(y_sample, y_pred))
        mcc_scores.append(matthews_corrcoef(y_sample, y_pred))
        roc_scores.append(roc_auc_score(y_sample, y_pred_prob))
        pr_scores.append(average_precision_score(y_sample, y_pred_prob))

    # metrics on full evaluation set
    y_pred_full      = model.predict(X_eval)
    y_pred_prob_full = model.predict_proba(X_eval)[:, 1]

    # print results
    print(f"Accuracy : {accuracy_score(y_eval, y_pred_full):.4f}  (95% CI: {np.percentile(acc_scores, 2.5):.4f} - {np.percentile(acc_scores, 97.5):.4f})")
    print(f"F1       : {f1_score(y_eval, y_pred_full):.4f}  (95% CI: {np.percentile(f1_scores, 2.5):.4f} - {np.percentile(f1_scores, 97.5):.4f})")
    print(f"MCC      : {matthews_corrcoef(y_eval, y_pred_full):.4f}  (95% CI: {np.percentile(mcc_scores, 2.5):.4f} - {np.percentile(mcc_scores, 97.5):.4f})")
    print(f"ROC-AUC  : {roc_auc_score(y_eval, y_pred_prob_full):.4f}  (95% CI: {np.percentile(roc_scores, 2.5):.4f} - {np.percentile(roc_scores, 97.5):.4f})")
    print(f"PR-AUC   : {average_precision_score(y_eval, y_pred_prob_full):.4f}  (95% CI: {np.percentile(pr_scores, 2.5):.4f} - {np.percentile(pr_scores, 97.5):.4f})")

    return {
    'acc_scores': acc_scores,
    'f1_scores' : f1_scores,
    'mcc_scores': mcc_scores,
    'roc_scores': roc_scores,
    'pr_scores' : pr_scores
}
    
   
