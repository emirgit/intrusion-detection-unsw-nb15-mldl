"""
IoT Network Intrusion Detection System - UNSW-NB15
Machine Learning Pipeline (Binary + Multi-class Classification)
OPTIMIZED: RF Feature Importance + Hyperparameter Tuning
SPLIT: Orijinal training-set / testing-set parquet dosyaları kullanılıyor
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics, preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 0) Proje kök dizini & alt dizinler
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# ============================================================
# 1) VERİ YÜKLEME (Train / Test ayrı)
# ============================================================
print("=" * 60)
print("[1] Veri yükleniyor (orijinal train/test split)...")
print("=" * 60)

train_raw = pd.read_parquet(os.path.join(DATASET_DIR, "UNSW_NB15_training-set.parquet"))
test_raw = pd.read_parquet(os.path.join(DATASET_DIR, "UNSW_NB15_testing-set.parquet"))

print(f"Training set : {train_raw.shape}")
print(f"Testing set  : {test_raw.shape}")


# ============================================================
# 2) ÖN İŞLEME FONKSİYONU (her iki sete aynı şekilde uygula)
# ============================================================
def preprocess(df):
    """Categorical -> str, service '-' -> NaN, drop NaN, tip dönüşümü."""
    df = df.copy()
    for c in df.select_dtypes(include=["category"]).columns:
        df[c] = df[c].astype(str)
    df["service"] = df["service"].replace("-", np.nan)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    for c in df.select_dtypes(include=["object"]).columns:
        if c not in ["proto", "service", "state", "attack_cat"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


print("=" * 60)
print("[2] Veri ön işleme...")
print("=" * 60)

train_data = preprocess(train_raw)
test_data = preprocess(test_raw)

print(f"Train (ön işleme sonrası): {train_data.shape}")
print(f"Test  (ön işleme sonrası): {test_data.shape}")
print(f"\nTrain attack dağılımı:\n{train_data['attack_cat'].value_counts()}\n")

# ============================================================
# 3) GÖRSELLEŞTİRME (train verisi üzerinden)
# ============================================================
print("=" * 60)
print("[3] Görselleştirme...")
print("=" * 60)

all_data = pd.concat([train_data, test_data], ignore_index=True)

plt.figure(figsize=(8, 8))
plt.pie(all_data.label.value_counts(), labels=["normal", "abnormal"], autopct="%0.2f%%")
plt.title("Pie chart distribution of normal and abnormal labels", fontsize=16)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "Pie_chart_binary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> plots/Pie_chart_binary.png")

plt.figure(figsize=(8, 8))
ac = all_data.attack_cat.value_counts()
plt.pie(ac, labels=ac.index, autopct="%0.2f%%")
plt.title("Pie chart distribution of multi-class labels")
plt.legend(loc="best")
plt.savefig(os.path.join(PLOTS_DIR, "Pie_chart_multi.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> plots/Pie_chart_multi.png")

del all_data  # bellek tasarrufu

# ============================================================
# 4) ONE-HOT ENCODING (tutarlı sütunlar)
# ============================================================
print("=" * 60)
print("[4] One-hot encoding...")
print("=" * 60)

cat_col = ["proto", "service", "state"]
print(f"Kategorik sütunlar: {cat_col}")

train_data = pd.get_dummies(train_data, columns=cat_col)
test_data = pd.get_dummies(test_data, columns=cat_col)

# Sütunları hizala — train'de olup test'te olmayan sütunlara 0, fazlalıkları sil
train_cols = set(train_data.columns)
test_cols = set(test_data.columns)

for c in train_cols - test_cols:
    test_data[c] = 0
for c in test_cols - train_cols:
    train_data[c] = 0

# Aynı sırayla hizala
test_data = test_data[train_data.columns]

print(f"One-hot sonrası sütun: {train_data.shape[1]}")

# ============================================================
# 5) NORMALİZASYON (train'e fit, test'e transform)
# ============================================================
print("=" * 60)
print("[5] Normalizasyon (MinMaxScaler)...")
print("=" * 60)

num_col = list(train_data.select_dtypes(include="number").columns)
if "id" in num_col:
    num_col.remove("id")
num_col.remove("label")

scaler = MinMaxScaler(feature_range=(0, 1))
train_data[num_col] = scaler.fit_transform(train_data[num_col])
test_data[num_col] = scaler.transform(test_data[num_col])

print("Normalizasyon tamamlandı (fit: train, transform: train+test).\n")

# ============================================================
# 6) LABEL ENCODING
# ============================================================
print("=" * 60)
print("[6] Label Encoding...")
print("=" * 60)

le1 = preprocessing.LabelEncoder()
le1.fit(["abnormal", "normal"])
print(f"Binary sınıflar: {le1.classes_}")
np.save(os.path.join(BASE_DIR, "le1_classes.npy"), le1.classes_, allow_pickle=True)

le2 = preprocessing.LabelEncoder()
le2.fit(sorted(train_data["attack_cat"].unique()))
print(f"Multi-class sınıflar: {le2.classes_}")
np.save(os.path.join(BASE_DIR, "le2_classes.npy"), le2.classes_, allow_pickle=True)

# --- Binary datasets ---
def make_binary_label(df):
    return le1.transform(df["label"].map(lambda x: "normal" if x == 0 else "abnormal"))

train_bin = train_data.copy()
train_bin["label"] = make_binary_label(train_bin)
test_bin = test_data.copy()
test_bin["label"] = make_binary_label(test_bin)

# --- Multi-class datasets ---
train_multi = train_data.copy()
train_multi_label = le2.transform(train_multi["attack_cat"])
# attack_cat one-hot'a çevrilmeden drop et
train_multi = train_multi.drop(columns=["attack_cat"])
train_multi["label"] = train_multi_label

test_multi = test_data.copy()
test_multi_label = le2.transform(test_multi["attack_cat"])
test_multi = test_multi.drop(columns=["attack_cat"])
test_multi["label"] = test_multi_label

# Binary'den de attack_cat kaldır
if "attack_cat" in train_bin.columns:
    train_bin = train_bin.drop(columns=["attack_cat"])
    test_bin = test_bin.drop(columns=["attack_cat"])

# ============================================================
# 7) FEATURE SELECTION - RF Feature Importance + Korelasyon
# ============================================================
print("=" * 60)
print("[7] Feature Selection (RF Importance + Korelasyon hibrit)...")
print("=" * 60)

# --- Binary ---
bin_num_col = list(train_bin.select_dtypes(include="number").columns)

plt.figure(figsize=(20, 8))
corr_bin = train_bin[bin_num_col].corr()
sns.heatmap(corr_bin, vmax=1.0, annot=False)
plt.title("Correlation Matrix for Binary Labels", fontsize=16)
plt.savefig(os.path.join(PLOTS_DIR, "correlation_matrix_bin.png"), dpi=150, bbox_inches="tight")
plt.close()

corr_ybin = abs(corr_bin["label"])
corr_features_bin = set(corr_ybin[corr_ybin > 0.1].index) - {"label"}

X_temp = train_bin.drop(columns=["label"])
y_temp = train_bin["label"]
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X_temp, y_temp)
importances = pd.Series(rf_temp.feature_importances_, index=X_temp.columns)
rf_features_bin = set(importances.nlargest(20).index)

bin_selected = sorted(list(corr_features_bin | rf_features_bin))
print(f"Binary - Seçilen feature sayısı: {len(bin_selected)}")
print(f"  Features: {bin_selected}")

plt.figure(figsize=(12, 6))
importances.nlargest(20).plot(kind="barh")
plt.title("Top 20 Features - Binary (RF Importance)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance_bin.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> plots/feature_importance_bin.png")

X_train_bin = train_bin[bin_selected]
y_train_bin = train_bin["label"]
X_test_bin = test_bin[bin_selected]
y_test_bin = test_bin["label"]

# --- Multi-class ---
multi_num_col = list(train_multi.select_dtypes(include="number").columns)

plt.figure(figsize=(20, 8))
corr_multi = train_multi[multi_num_col].corr()
sns.heatmap(corr_multi, vmax=1.0, annot=False)
plt.title("Correlation Matrix for Multi Labels", fontsize=16)
plt.savefig(os.path.join(PLOTS_DIR, "correlation_matrix_multi.png"), dpi=150, bbox_inches="tight")
plt.close()

corr_ymulti = abs(corr_multi["label"])
corr_features_multi = set(corr_ymulti[corr_ymulti > 0.1].index) - {"label"}
corr_features_multi = {f for f in corr_features_multi if not f.startswith("attack_cat_")}

leak_cols = [c for c in train_multi.columns if c.startswith("attack_cat_")]
X_temp = train_multi.drop(columns=["label"] + leak_cols)
y_temp = train_multi["label"]
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X_temp, y_temp)
importances_multi = pd.Series(rf_temp.feature_importances_, index=X_temp.columns)
rf_features_multi = set(importances_multi.nlargest(20).index)

multi_selected = sorted(list(corr_features_multi | rf_features_multi))
print(f"\nMulti-class - Seçilen feature sayısı: {len(multi_selected)}")
print(f"  Features: {multi_selected}")

plt.figure(figsize=(12, 6))
importances_multi.nlargest(20).plot(kind="barh")
plt.title("Top 20 Features - Multi-class (RF Importance)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance_multi.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> plots/feature_importance_multi.png")

X_train_multi = train_multi[multi_selected]
y_train_multi = train_multi["label"]
X_test_multi = test_multi[multi_selected]
y_test_multi = test_multi["label"]

# Prepared datasets kaydet
pd.concat([X_train_bin, y_train_bin], axis=1).to_csv(os.path.join(DATASET_DIR, "bin_data.csv"), index=False)
pd.concat([X_train_multi, y_train_multi], axis=1).to_csv(os.path.join(DATASET_DIR, "multi_data.csv"), index=False)


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def evaluate_model(y_test, y_pred, model_name, label_classes=None):
    print(f"\n  --- {model_name} Sonuçları ---")
    print(f"  Mean Absolute Error  : {metrics.mean_absolute_error(y_test, y_pred):.6f}")
    print(f"  Mean Squared Error   : {metrics.mean_squared_error(y_test, y_pred):.6f}")
    print(f"  RMSE                 : {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.6f}")
    print(f"  R2 Score             : {metrics.explained_variance_score(y_test, y_pred) * 100:.4f}%")
    print(f"  Accuracy             : {accuracy_score(y_test, y_pred) * 100:.4f}%")
    if label_classes is not None:
        print(classification_report(y_test, y_pred, target_names=label_classes, zero_division=0))


def plot_predictions(y_test, y_pred, title, filename, start=0, end=200):
    plt.figure(figsize=(20, 8))
    plt.plot(y_pred[start:end], label="prediction", linewidth=2.0, color="blue")
    plt.plot(y_test[start:end].values if hasattr(y_test, "values") else y_test[start:end],
             label="real_values", linewidth=2.0, color="lightcoral")
    plt.legend(loc="best")
    plt.title(title)
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> plots/{filename}")


def save_model(model, filename):
    filepath = os.path.join(MODELS_DIR, filename)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"  -> Model: {filepath}")


def save_predictions(y_test, y_pred, filename):
    df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    df.to_csv(os.path.join(PREDICTIONS_DIR, filename), index=False)
    print(f"  -> Tahminler: predictions/{filename}")


# ############################################################
#                  BINARY CLASSIFICATION
# ############################################################
print("\n" + "=" * 60)
print("        BINARY CLASSIFICATION")
print("=" * 60)

print(f"  Train: {X_train_bin.shape}, Test: {X_test_bin.shape}\n")

# -------------------------------------------------------
# 1) Linear Regression (Binary)
# -------------------------------------------------------
print("[Binary] Linear Regression eğitiliyor...")
lr_bin = LinearRegression()
lr_bin.fit(X_train_bin, y_train_bin)
y_pred_raw = lr_bin.predict(X_test_bin)
y_pred = np.where(y_pred_raw > 0.5, 1, 0)

evaluate_model(y_test_bin, y_pred, "Linear Regression (Binary)", le1.classes_)
plot_predictions(y_test_bin, y_pred, "Linear Regression Binary", "lr_real_pred_bin.png")
save_predictions(y_test_bin, y_pred, "lr_real_pred_bin.csv")
save_model(lr_bin, "linear_regressor_binary.pkl")

# -------------------------------------------------------
# 2) Logistic Regression (Binary) + Tuning
# -------------------------------------------------------
print("\n[Binary] Logistic Regression (tuning)...")
logr_search = RandomizedSearchCV(
    LogisticRegression(random_state=123, max_iter=5000),
    {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"]},
    n_iter=8, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
)
logr_search.fit(X_train_bin, y_train_bin)
logr_bin = logr_search.best_estimator_
print(f"  Best params: {logr_search.best_params_}")
y_pred = logr_bin.predict(X_test_bin)

evaluate_model(y_test_bin, y_pred, "Logistic Regression (Binary)", le1.classes_)
plot_predictions(y_test_bin, y_pred, "Logistic Regression Binary", "logr_real_pred_bin.png")
save_predictions(y_test_bin, y_pred, "logr_real_pred_bin.csv")
save_model(logr_bin, "logistic_regressor_binary.pkl")

# -------------------------------------------------------
# 3) SVM (Binary)
# -------------------------------------------------------
print("\n[Binary] SVM eğitiliyor (C=10, rbf)...")
lsvm_bin = SVC(kernel="rbf", C=10, gamma="auto")
lsvm_bin.fit(X_train_bin, y_train_bin)
print(f"  Params: kernel=rbf, C=10")
y_pred = lsvm_bin.predict(X_test_bin)

evaluate_model(y_test_bin, y_pred, "SVM (Binary)", le1.classes_)
plot_predictions(y_test_bin, y_pred, "SVM Binary", "lsvm_real_pred_bin.png")
save_predictions(y_test_bin, y_pred, "lsvm_real_pred_bin.csv")
save_model(lsvm_bin, "lsvm_binary.pkl")

# -------------------------------------------------------
# 4) KNN (Binary) + Tuning
# -------------------------------------------------------
print("\n[Binary] KNN (tuning)...")
knn_search = RandomizedSearchCV(
    KNeighborsClassifier(),
    {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"],
     "metric": ["minkowski", "manhattan"]},
    n_iter=10, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
)
knn_search.fit(X_train_bin, y_train_bin)
knn_bin = knn_search.best_estimator_
print(f"  Best params: {knn_search.best_params_}")
y_pred = knn_bin.predict(X_test_bin)

evaluate_model(y_test_bin, y_pred, "KNN (Binary)", le1.classes_)
plot_predictions(y_test_bin, y_pred, "KNN Binary", "knn_real_pred_bin.png")
save_predictions(y_test_bin, y_pred, "knn_real_pred_bin.csv")
save_model(knn_bin, "knn_binary.pkl")

# -------------------------------------------------------
# 5) Random Forest (Binary) + Tuning
# -------------------------------------------------------
print("\n[Binary] Random Forest (tuning)...")
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=123, n_jobs=-1),
    {"n_estimators": [100, 200, 300], "max_depth": [10, 20, 30, None],
     "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
    n_iter=15, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
)
rf_search.fit(X_train_bin, y_train_bin)
rf_bin = rf_search.best_estimator_
print(f"  Best params: {rf_search.best_params_}")
y_pred = rf_bin.predict(X_test_bin)

evaluate_model(y_test_bin, y_pred, "Random Forest (Binary)", le1.classes_)
plot_predictions(y_test_bin, y_pred, "Random Forest Binary", "rf_real_pred_bin.png", 200, 400)
save_predictions(y_test_bin, y_pred, "rf_real_pred_bin.csv")
save_model(rf_bin, "random_forest_binary.pkl")

# -------------------------------------------------------
# 6) Decision Tree (Binary) + Tuning
# -------------------------------------------------------
print("\n[Binary] Decision Tree (tuning)...")
dt_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=123),
    {"max_depth": [5, 10, 20, 30, None], "min_samples_split": [2, 5, 10],
     "min_samples_leaf": [1, 2, 4], "criterion": ["gini", "entropy"]},
    n_iter=15, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
)
dt_search.fit(X_train_bin, y_train_bin)
dt_bin = dt_search.best_estimator_
print(f"  Best params: {dt_search.best_params_}")
y_pred = dt_bin.predict(X_test_bin)

evaluate_model(y_test_bin, y_pred, "Decision Tree (Binary)", le1.classes_)
plot_predictions(y_test_bin, y_pred, "Decision Tree Binary", "dt_real_pred_bin.png", 300, 500)
save_predictions(y_test_bin, y_pred, "dt_real_pred_bin.csv")
save_model(dt_bin, "decision_tree_binary.pkl")


# ############################################################
#               MULTI-CLASS CLASSIFICATION
# ############################################################
print("\n" + "=" * 60)
print("        MULTI-CLASS CLASSIFICATION")
print("=" * 60)

print(f"  Train: {X_train_multi.shape}, Test: {X_test_multi.shape}\n")

# -------------------------------------------------------
# 1) Linear Regression (Multi)
# -------------------------------------------------------
print("[Multi] Linear Regression eğitiliyor...")
lr_multi = LinearRegression()
lr_multi.fit(X_train_multi, y_train_multi)
y_pred_raw = lr_multi.predict(X_test_multi)
y_pred = np.clip(np.round(y_pred_raw).astype(int), 0, len(le2.classes_) - 1)

evaluate_model(y_test_multi, y_pred, "Linear Regression (Multi)", le2.classes_)
plot_predictions(y_test_multi, y_pred, "Linear Regression Multi-class", "lr_real_pred_multi.png", 100, 200)
save_predictions(y_test_multi, y_pred, "lr_real_pred_multi.csv")
save_model(lr_multi, "linear_regressor_multi.pkl")

# -------------------------------------------------------
# 2) Logistic Regression (Multi) + Tuning
# -------------------------------------------------------
print("\n[Multi] Logistic Regression (tuning)...")
logr_search_m = RandomizedSearchCV(
    LogisticRegression(random_state=123, max_iter=5000),
    {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "newton-cg"]},
    n_iter=6, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
)
logr_search_m.fit(X_train_multi, y_train_multi)
logr_multi = logr_search_m.best_estimator_
print(f"  Best params: {logr_search_m.best_params_}")
y_pred = logr_multi.predict(X_test_multi)

evaluate_model(y_test_multi, y_pred, "Logistic Regression (Multi)", le2.classes_)
plot_predictions(y_test_multi, y_pred, "Logistic Regression Multi-class", "logr_real_pred_multi.png")
save_predictions(y_test_multi, y_pred, "logr_real_pred_multi.csv")
save_model(logr_multi, "logistic_regressor_multi.pkl")

# -------------------------------------------------------
# 3) SVM (Multi)
# -------------------------------------------------------
print("\n[Multi] SVM eğitiliyor (C=10, linear)...")
lsvm_multi = SVC(kernel="linear", C=10, gamma="auto")
lsvm_multi.fit(X_train_multi, y_train_multi)
print(f"  Params: kernel=linear, C=10")
y_pred = lsvm_multi.predict(X_test_multi)

evaluate_model(y_test_multi, y_pred, "SVM (Multi)", le2.classes_)
plot_predictions(y_test_multi, y_pred, "SVM Multi-class", "lsvm_real_pred_multi.png")
save_predictions(y_test_multi, y_pred, "lsvm_real_pred_multi.csv")
save_model(lsvm_multi, "lsvm_multi.pkl")

# -------------------------------------------------------
# 4) KNN (Multi) + Tuning
# -------------------------------------------------------
print("\n[Multi] KNN (tuning)...")
knn_search_m = RandomizedSearchCV(
    KNeighborsClassifier(),
    {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"],
     "metric": ["minkowski", "manhattan"]},
    n_iter=10, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
)
knn_search_m.fit(X_train_multi, y_train_multi)
knn_multi = knn_search_m.best_estimator_
print(f"  Best params: {knn_search_m.best_params_}")
y_pred = knn_multi.predict(X_test_multi)

evaluate_model(y_test_multi, y_pred, "KNN (Multi)", le2.classes_)
plot_predictions(y_test_multi, y_pred, "KNN Multi-class", "knn_real_pred_multi.png", 400, 500)
save_predictions(y_test_multi, y_pred, "knn_real_pred_multi.csv")
save_model(knn_multi, "knn_multi.pkl")

# -------------------------------------------------------
# 5) Random Forest (Multi) + Tuning
# -------------------------------------------------------
print("\n[Multi] Random Forest (tuning)...")
rf_search_m = RandomizedSearchCV(
    RandomForestClassifier(random_state=50, n_jobs=-1),
    {"n_estimators": [100, 200, 300], "max_depth": [10, 20, 30, None],
     "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
     "class_weight": ["balanced", "balanced_subsample", None]},
    n_iter=20, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
)
rf_search_m.fit(X_train_multi, y_train_multi)
rf_multi = rf_search_m.best_estimator_
print(f"  Best params: {rf_search_m.best_params_}")
y_pred = rf_multi.predict(X_test_multi)

evaluate_model(y_test_multi, y_pred, "Random Forest (Multi)", le2.classes_)
plot_predictions(y_test_multi, y_pred, "Random Forest Multi-class", "rf_real_pred_multi.png", 500, 600)
save_predictions(y_test_multi, y_pred, "rf_real_pred_multi.csv")
save_model(rf_multi, "random_forest_multi.pkl")

# -------------------------------------------------------
# 6) Decision Tree (Multi) + Tuning
# -------------------------------------------------------
print("\n[Multi] Decision Tree (tuning)...")
dt_search_m = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=123),
    {"max_depth": [5, 10, 20, 30, None], "min_samples_split": [2, 5, 10],
     "min_samples_leaf": [1, 2, 4], "criterion": ["gini", "entropy"],
     "class_weight": ["balanced", None]},
    n_iter=15, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
)
dt_search_m.fit(X_train_multi, y_train_multi)
dt_multi = dt_search_m.best_estimator_
print(f"  Best params: {dt_search_m.best_params_}")
y_pred = dt_multi.predict(X_test_multi)

evaluate_model(y_test_multi, y_pred, "Decision Tree (Multi)", le2.classes_)
plot_predictions(y_test_multi, y_pred, "Decision Tree Multi-class", "dt_real_pred_multi.png", 400, 700)
save_predictions(y_test_multi, y_pred, "dt_real_pred_multi.csv")
save_model(dt_multi, "decision_tree_multi.pkl")


# ============================================================
# SONUÇ ÖZET
# ============================================================
print("\n" + "=" * 60)
print("        TAMAMLANDI!")
print("=" * 60)
print(f"""
Veri Bölünmesi:
  Training set: {train_raw.shape[0]} satır (UNSW_NB15_training-set.parquet)
  Testing set : {test_raw.shape[0]} satır (UNSW_NB15_testing-set.parquet)

Optimizasyonlar:
  [+] Orijinal Split   : Dataset'in kendi train/test bölünmesi kullanıldı
  [+] Feature Selection: Korelasyon (>0.1) + RF Feature Importance (Top 20) hibrit
  [+] Hyperparameter   : RandomizedSearchCV (3-fold CV) uygulandı
  [+] Scaler           : Train'e fit, test'e transform (no leakage)

Çıktılar:
  plots/       -> Grafikler + feature importance plot'ları
  models/      -> Optimize edilmiş modeller (pickle)
  predictions/ -> Gerçek vs Tahmin CSV dosyaları
  dataset/     -> Ön işlenmiş bin_data.csv ve multi_data.csv
""")