from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import sys
import pandas as pd
import joblib
import os
import time


def main():
    
    if len(sys.argv) != 2:
        print("Usage rftrain [csv]")
        return
    
    print("Leyendo Dataset...")
    df = pd.read_csv(sys.argv[1])
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
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    
    print("--- Buscando los mejores parámetros ---")
    start = time.time()
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
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_test, y_pred))
    
if __name__ == "__main__":
    main()
    
    