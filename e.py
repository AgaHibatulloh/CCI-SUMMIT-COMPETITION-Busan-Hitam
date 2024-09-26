import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
from scipy.stats import randint
from tqdm import tqdm  # Import tqdm untuk progress bar

warnings.filterwarnings("ignore")

# Path ke file CSV di desktop
train_path = 'C:/Users/agahi/Desktop/train.csv'
test_path = 'C:/Users/agahi/Desktop/test.csv'

# Baca data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Preprocessing
train.columns = train.columns.str.strip().str.replace(' ', '')
train["income"] = train["income"].str.strip()
test.columns = test.columns.str.strip().str.replace(' ', '')

train = train.replace('?', np.nan)
test = test.replace('?', np.nan)
train.dropna(inplace=True)

cat_columns = ['KelasPekerjaan', 'Pendidikan', 'JenjangPendidikan', 'Status', 'Pekerjaan', 'Hubungan', 'Etnis', 'sex', 'AsalNegara']
df_dumy_train = pd.get_dummies(train, columns=cat_columns)
df_dumy_test = pd.get_dummies(test, columns=cat_columns)

df_dumy_test = df_dumy_test.reindex(columns=df_dumy_train.columns, fill_value=0)

X = df_dumy_train.drop("income", axis=1)
y = df_dumy_train["income"].apply(lambda x: 1 if x == '>50K' else 0)

# Oversampling dengan SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Membagi data
X_train, X_val, y_train, y_val = train_test_split(X_smote, y_smote, test_size=0.1, random_state=101)

# Scaling fitur
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_val = scaler.transform(X_val)
scaled_X_test = scaler.transform(df_dumy_test.drop("income", axis=1, errors='ignore'))

# Tuning Hyperparameter dengan RandomizedSearchCV untuk RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
param_dist_rf = {
    'n_estimators': randint(100, 1000),  # Meningkatkan range n_estimators
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': randint(10, 150),  # Meningkatkan range max_depth
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False]
}

# Tambahkan progress bar dengan tqdm
random_search_rf = RandomizedSearchCV(
    rf, param_distributions=param_dist_rf, 
    n_iter=200, cv=10, random_state=42, n_jobs=-1, verbose=1  # Meningkatkan n_iter dan cv
)

# Progress tracking saat training model
print("Training model dengan RandomizedSearchCV...")
with tqdm(total=200) as pbar:  # Meningkatkan total iterasi
    random_search_rf.fit(scaled_X_train, y_train)
    pbar.update(200)

best_rf = random_search_rf.best_estimator_

# Fine-tuning dengan GridSearchCV setelah RandomizedSearchCV
param_grid_rf = {
    'n_estimators': [best_rf.n_estimators - 50, best_rf.n_estimators, best_rf.n_estimators + 50],
    'max_depth': [best_rf.max_depth - 10, best_rf.max_depth, best_rf.max_depth + 10],
    'min_samples_split': [best_rf.min_samples_split - 2, best_rf.min_samples_split, best_rf.min_samples_split + 2],
    'min_samples_leaf': [best_rf.min_samples_leaf - 1, best_rf.min_samples_leaf, best_rf.min_samples_leaf + 1]
}

grid_search_rf = GridSearchCV(best_rf, param_grid=param_grid_rf, cv=10, n_jobs=-1, verbose=2)
grid_search_rf.fit(scaled_X_train, y_train)

# Ambil model terbaik setelah GridSearchCV
best_rf = grid_search_rf.best_estimator_

# Evaluasi performa model pada validation set
val_predictions = best_rf.predict(scaled_X_val)
print("F1 Score (Validation Set):", f1_score(y_val, val_predictions))
print(classification_report(y_val, val_predictions))

ConfusionMatrixDisplay.from_estimator(best_rf, scaled_X_val, y_val)
plt.show()

# Prediksi pada data test dengan progress bar
print("Melakukan prediksi pada data test...")
with tqdm(total=len(scaled_X_test)) as pbar:
    test_predictions = best_rf.predict(scaled_X_test)
    pbar.update(len(scaled_X_test))

# Buat DataFrame untuk submission
submission = pd.DataFrame({
    'ID': test['ID'],
    'income': ['1' if pred == 1 else 0 for pred in test_predictions]
})

submission.to_csv('submissionE.csv', index=False)
print("Submission file created: 'submissionE.csv'")
