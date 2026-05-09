from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import sys
import pandas as pd
import joblib
import os
import time
import argparse
from imblearn.over_sampling import SMOTE, SMOTENC
import optuna
import generate_joblib_folder
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.pipeline import Pipeline

def balanceo_cfm(X_train, y_train, tarea):
    if tarea == '8clases':
        smt_strategy = {
            1: 10000,
            5: 10000
        }
        rus_strategy = {
            0: 10000,
            2: 10000,
            3: 10000,
            4: 10000,
            6: 10000,
            7: 10000
        }
        smote = SMOTENC(sampling_strategy=smt_strategy, random_state=42, k_neighbors=5, categorical_features=['Dst Port', 'Protocol', 'Fwd PSH Flags'])
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        rus = RandomUnderSampler(sampling_strategy=rus_strategy, random_state=42)
        X_train_bal, y_train_bal = rus.fit_resample(X_train_bal, y_train_bal)
    elif tarea == '2clases':
        rus = RandomUnderSampler()
        X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
    else:
        target_samples = 250
        counts = y_train.value_counts()
        strategy_rus = {label: min(count, target_samples) for label, count in counts.items()}
        rus = RandomUnderSampler(sampling_strategy=strategy_rus, random_state=42)
        X_rus, y_rus = rus.fit_resample(X_train, y_train)
        strategy_smote = {label: target_samples for label in y_rus.unique()}
        smote_nc = SMOTENC(categorical_features=['Dst Port', 'Protocol', 'Fwd PSH Flags'], 
                   sampling_strategy=strategy_smote, 
                   random_state=42)
        X_train_bal, y_train_bal = smote_nc.fit_resample(X_rus, y_rus)
        
    print(f"Valores de cada clase: {y_train_bal.value_counts()}")
    return (X_train_bal, y_train_bal)

def balanceo_cfm2(X_train, y_train):
    rus = RandomUnderSampler()
    return rus.fit_resample(X_train, y_train)


def process(input, folder, tarea):
    
    print("Leyendo Dataset...")
    df = pd.read_csv(input)
    if df.size == 0:
        print("Error reading csv")
        return
    
    print(f"Dataset {input} cargado")
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    """
    print("Realizando undersampling...")
    if tarea != '60clases':
        enn_1s = EditedNearestNeighbours(sampling_strategy=[0], n_neighbors=3)
    else:
        enn_1s = EditedNearestNeighbours(sampling_strategy=[4], n_neighbors=3)
    X_train, y_train = enn_1s.fit_resample(X_train, y_train)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    #X_train, y_train = balanceo_cfm(X_train, y_train, tarea)
    X_train_scaled, y_train = balanceo_cfm(X_train_scaled, y_train, tarea)
    
    

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "kNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB()
    }

    search_grid = {
        "RandomForest": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [10, 20, 40, None],
            "min_samples_split": [2, 5, None]
        },
        "kNN": {
            "n_neighbors": [3, 5, 8, 11],
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
        
        
        
    def objetive(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_l{i}", 32, 256, log=True))
        
        params = {
            'hidden_layer_sizes': tuple(layers),
            'activation': trial.suggest_categorical("activation", ["relu", "tanh"]),
            'solver': 'adam',
            'alpha': trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            'learning_rate_init': trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            'max_iter': 500,
            'random_state': 42
        }
        
        model = MLPClassifier(**params)
        
        score = cross_val_score(model, X_train_scaled, y_train, cv=3, n_jobs=-1, scoring='f1_macro').mean()
        
        return score
    
    def objective_exhaustive(trial):
        n_layers = trial.suggest_int("n_layers", 1, 4)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_l{i}", 32, 512, log=True))
        
        solver = trial.suggest_categorical("solver", ["adam", "sgd"])

        params = {
            'hidden_layer_sizes': tuple(layers),
            'activation': trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
            'solver': solver,
            'alpha': trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True),
            'learning_rate': trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
            'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128, 256, "auto"]),
            'max_iter': 1000, # Aumentamos para asegurar convergencia en redes profundas
            'early_stopping': True, # Evita que la red aprenda de memoria (overfitting)
            'validation_fraction': 0.1,
            'random_state': 42
        }
        
        model = MLPClassifier(**params)
        
        # Usamos f1_macro porque para 60 clases el accuracy es una métrica "mentirosa"
        score = cross_val_score(model, X_train_scaled, y_train, cv=3, n_jobs=-1, scoring='f1_macro').mean()
        
        return score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_exhaustive, n_trials=30)
    
    best_params = study.best_params
    
    layers = [best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])]

    final_model = MLPClassifier(
        hidden_layer_sizes=tuple(layers),
        activation=best_params["activation"],
        alpha=best_params["alpha"],
        learning_rate_init=best_params["learning_rate_init"],
        max_iter=500,
        random_state=42
    )    
    final_model.fit(X_train_scaled, y_train)
    mejores_modelos["MLP"] = final_model
    
        
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Archivo csv para el entrenamiento y validacion")
    arguments = parser.parse_args()
    folder = '09-05-2026-CFM-NobalanCleanedRFBalancedSCALED'
    generate_joblib_folder.create(folder)
    
    
    tareas = {'2clases': 'label1', '8clases': 'label2', '60clases': 'label3'}
    for tarea in tareas.items():
        process(f"csv/{tarea[1]}_{arguments.input}", f"joblibs/{folder}/{tarea[0]}", tarea[0])