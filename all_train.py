from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import sys
import pandas as pd
import joblib
import os
import time
import argparse
from imblearn.over_sampling import SMOTE


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Archivo csv para el entrenamiento y validacion")
    arguments = parser.parse_args()
    
    print("Leyendo Dataset...")
    df = pd.read_csv(arguments.input)
    if df.size == 0:
        print("Error reading csv")
        return
    
    print(f"Dataset {arguments.input} cargado")
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    
    
    

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "kNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(random_state=42, early_stopping=True),
        "NaiveBayes": GaussianNB()
    }

    search_grid = {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5]
        },
        "MLP": {
            "hidden_layer_sizes": [(50, 50), (100,)],
            "activation": ['tanh', 'relu'],
            "learning_rate_init": [0.001, 0.01],
            "max_iter": [200, 500]
        },
        "kNN": {
            "n_neighbors": [3, 5, 11],
            "weights": ['uniform', 'distance']
        },
        "NaiveBayes": {
            "var_smoothing": [1e-9, 1e-8]
        }
    }

    mejores_modelos = {}

    for nombre in models:
        print(f"Ajustando hiperparámetros para: {nombre}")
        search = RandomizedSearchCV(
            estimator=models[nombre],
            param_distributions=search_grid[nombre],
            n_iter=7,
            cv=3,
            n_jobs=2,
            random_state=42,
            verbose=1
        )
        search.fit(X_train_scaled, y_train)
        
        mejores_modelos[nombre] = search.best_estimator_
        print(f"Mejores parámetros para {nombre}: {search.best_params_}")
        
    folder = 'joblibs'
    if not os.path.exists(folder):
        os.makedirs(folder)
    joblib.dump(scaler, os.path.join(folder, 'scaler_final.joblib'))
    joblib.dump(X.columns.tolist(), os.path.join(folder, 'features_list.joblib'))
    for model in mejores_modelos:
        joblib.dump(mejores_modelos[model], os.path.join(folder, f'{model}.joblib'))
        
    for nombre, modelo in mejores_modelos.items():
        print(f"\n================ REPORTE: {nombre} ================")
        y_pred = modelo.predict(X_test_scaled)
        print(classification_report(y_test, y_pred))
        
if __name__ == "__main__":
    main()