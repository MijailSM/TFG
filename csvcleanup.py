import pandas as pd
import sys

def attack_bening_cleanup(df: pd.DataFrame):
    df['target'] = df['label1'].map({'benign': 0, 'attack': 1})
    df = df.drop(['device_name', 'device_mac', 'label_full','label2', 'label3', 'label4', 'timestamp'], axis=1)
    
    #Quitar columnas con datos iguales
    cols = [col for col in df.columns if df[col].nunique() <= 1]
    print(f"Se han borrado {len(cols)} columnas del DataFrame: \n{cols}")
    df = df.drop(cols, axis=1)
    df = df.select_dtypes(include=['number'])
    return df

def main():
    if len(sys.argv) != 2:
        print("Usage csvcleanup.py dataframename")
        return
    
    name = sys.argv[1]
    print("Leyendo csv...")
    df = pd.read_csv(name)
    if df.size == 0:
        print("Failed opening csv")
        return
    
    df = attack_bening_cleanup(df)
    print("Saving new DataFrame...")
    df.to_csv(f"NEW_{name}", index=False)
    return

if __name__ == "__main__":
    main()