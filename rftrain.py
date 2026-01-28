from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import pandas as pd
import joblib

def main():
    
    if len(sys.argv) != 2:
        print("Usage rftrain [csv]")
        return
    df = pd.read_csv(sys.argv[1])
    if df.size == 0:
        print("Error reading csv")
        return
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, 'rf_model_final.joblib')
    joblib.dump(scaler, 'scaler_final.joblib')
    joblib.dump(X.columns.tolist(), 'features_list.joblib')
    print("Entrenamiento completado y modelos guardados.")
    
    y_pred = model.predict(X_test_scaled)
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_test, y_pred))
    
if __name__ == "__main__":
    main()
    
    