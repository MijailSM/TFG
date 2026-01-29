import sys
import tarfile
import pandas as pd
import io

def mergecsvs(csvs: list) -> pd.DataFrame:
    
    print("Uniendo todo...")
    merged = pd.concat(csvs, ignore_index=True)
    
    merged = merged.sample(frac=1).reset_index(drop=True)
    
    print(f"porcentaje de cada clase:\n{merged['label1'].value_counts(normalize=True) * 100}")
    
    return merged

def attack_bening_cleanup(df: pd.DataFrame):
    df['target'] = df['label1'].map({'benign': 0, 'attack': 1})
    df = df.drop(['device_name', 'device_mac', 'label_full','label2', 'label3', 'label4', 'timestamp'], axis=1)
    
    #Quitar columnas con datos iguales
    cols = [col for col in df.columns if df[col].nunique() <= 1]
    print(f"Se han borrado {len(cols)} columnas con los mismos valores del DataFrame: \n{cols}")
    df = df.drop(cols, axis=1)
    df = df.select_dtypes(include=['number'])
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: mergecsv [name1] [name2] ... [namen]")
        return
    
    csvs = []
    for name in sys.argv[1:]:
        if name.endswith('.tar.xz'):
            print(f'Descomprimiendo {name}...')
            with tarfile.open(name, "r:xz") as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith('.csv'))
                f = tar.extractfile(member)
                df = pd.read_csv(io.BytesIO(f.read()))
        else:
            df = pd.read_csv(name)
        
        print(f"Cargando archivo: {name} || Columnas: {len(df.columns)}")
            
        csvs.append(df)
    
    merged = mergecsvs(csvs)
    print("DataFrame unido, limpiando...")
    merged = attack_bening_cleanup(merged)
    
    name = "Cleaned_DF.csv"
    print(f"Dataframe generado y limpio, guardando como {name}...")
    merged.to_csv(name, index=False)
    
if __name__ == "__main__":
    main()
    
    