from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
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
    parser.add_argument("-s", "--smote", action="store_true", help="Uso de SMOTE")
    arguments = parser.parse_args()
    
    print("Leyendo Dataset...")
    df = pd.read_csv(arguments.input)
    if df.size == 0:
        print("Error reading csv")
        return
    
    print(f"Dataset {sys.argv[1]} cargado")
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    if arguments.smote == True:
        print("USANDO SMOTE")
        smote = SMOTE()
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced',),
        param_grid=param_grid,
        cv=3,
        n_jobs=4,
        verbose=2
    )
    
    print("--- Buscando los mejores parámetros ---")
    start = time.time()
    if arguments.smote == True:
        grid_search.fit(X_train_smote, y_train_smote)
    else:
        grid_search.fit(X_train_scaled, y_train)
    end = time.time()

    # 4. Ver resultados
    print(f"Mejores parámetros: {grid_search.best_params_}")
    best_rf = grid_search.best_estimator_
    print(f"Tiempo de entrenamiento: {end - start} segundos")
    
    folder = 'joblibs'
    if not os.path.exists(folder):
        os.makedirs(folder)
    joblib.dump(best_rf, os.path.join(folder, 'rf_model_final.joblib'))
    joblib.dump(scaler, os.path.join(folder, 'scaler_final.joblib'))
    joblib.dump(X.columns.tolist(), os.path.join(folder, 'features_list.joblib'))
    print("Entrenamiento completado y modelos guardados.")
    
    y_pred = best_rf.predict(X_test_scaled)
    print(f"Puntuación de entrenamiento: {accuracy_score(y_train, best_rf.predict(X_train_scaled))}")
    print(f"Puntiación de test: {accuracy_score(y_test, y_pred)}")
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_test, y_pred))
    
if __name__ == "__main__":
    main()
    
    