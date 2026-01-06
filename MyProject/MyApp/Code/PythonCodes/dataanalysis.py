import numpy as np
import pandas as pd
import json
from scipy.stats import chi2_contingency, pearsonr

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = None
matrix = None
def analysis(context) :
    global matrix, df
    #print(context)
    df = pd.DataFrame(json.loads(context["df_data"]))
    print("Dtypes are ")
    print(df.dtypes)
    cols = df.columns
    matrix = pd.DataFrame(index=cols, columns=cols)
    matrix = mixed_correlation_matrix(df)
    

    target_col = df.columns[-1]

    correlations_with_target = matrix[target_col].drop(target_col).abs()

    # 🎯 Filter columns with correlation ≥ 0.5
    high_corr_features = correlations_with_target[correlations_with_target >= 0.0001].index.tolist()

    filtered_df = df[high_corr_features + [target_col]]

    print(f"🔍 Features with correlation ≥ 0.5 with target ({target_col}):")
    print(high_corr_features)
    print("🧾 Filtered DataFrame:")
    print(filtered_df)

    print("Filter Dtypes are ")
    print(filtered_df.dtypes)

    return filtered_df

    # return df


    # Only use numeric features again (in case some high-corr features are categorical)


def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def correlation_ratio(categories, values):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = values[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.mean(cat_measures)
    y_total_avg = np.sum(y_avg_array * n_array) / np.sum(n_array)
    numerator = np.sum(n_array * (y_avg_array - y_total_avg)**2)
    denominator = np.sum((values - y_total_avg)**2)
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta

def mixed_correlation_matrix(df):
    global matrix

    cols = df.columns
    matrix = pd.DataFrame(index=cols, columns=cols)

    for col1 in cols:
        for col2 in cols:
            if df[col1].dtype == 'object' and df[col2].dtype == 'object':
                matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
            elif df[col1].dtype != 'object' and df[col2].dtype != 'object':
                if df[col1].nunique() <= 1 or df[col2].nunique() <= 1:
                    matrix.loc[col1, col2] = 1 if col1 == col2 else 0
                else:
                    matrix.loc[col1, col2] = pearsonr(df[col1], df[col2])[0]
            elif df[col1].dtype == 'object' and df[col2].dtype != 'object':
                matrix.loc[col1, col2] = correlation_ratio(df[col1], df[col2])
            else:
                matrix.loc[col1, col2] = correlation_ratio(df[col2], df[col1])

    return matrix.astype(float)

def apply_pca(df, variance_threshold=0.95):
    # 1️⃣ Select numerical columns only
    num_df = df.select_dtypes(include=[np.number])

    # 2️⃣ Standardize numerical features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(num_df)

    # 3️⃣ Apply PCA
    pca = PCA(n_components=variance_threshold)
    pca_data = pca.fit_transform(scaled_data)

    # 4️⃣ Create DataFrame from PCA result
    pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(pca_data.shape[1])])

    # 5️⃣ Show variance explained
    print("✅ PCA Applied")
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance retained:", round(sum(pca.explained_variance_ratio_), 4))

    return pca_df


