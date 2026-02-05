import tarfile
import pandas as pd
import io
import os
import argparse
import ast

def mergecsvs(csvs: list) -> pd.DataFrame:
    
    print("Uniendo todo...")
    merged = pd.concat(csvs, ignore_index=True)
    
    merged = merged.sample(frac=1).reset_index(drop=True)
    
    print(f"porcentaje de cada clase:\n{merged['label1'].value_counts(normalize=True) * 100}")
    
    return merged

def protocol_extractor(df: pd.DataFrame) -> pd.DataFrame:
    df["protocols_list"] = df["network_protocols_all"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
    )
    target_protocols = ['tcp', 'udp', 'icmp', 'arp', 'tls', 'http', 'dns', 'mqtt', 'json']
    for protocol in target_protocols:
        df[f"proto_{protocol}"] = df["protocols_list"].apply(
            lambda x: 1 if protocol in x else 0
        )
    df = df.drop(["protocols_list", "network_protocols_all"], axis=1)
    print(f"Se han procesado los siguientes protocolos: {target_protocols}")
    return df

def attack_bening_basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df['target'] = df['label1'].map({'benign': 0, 'attack': 1})
    df = df.drop(['device_name', 'device_mac', 'label_full','label2', 'label3', 'label4', 'timestamp'], axis=1)
    
    #Quitar columnas con datos iguales
    cols = [col for col in df.columns if df[col].nunique() <= 1]
    print(f"Se han borrado {len(cols)} columnas con los mismos valores del DataFrame: \n{cols}")
    df = df.drop(cols, axis=1)
    df = df.select_dtypes(include=['number'])
    return df

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", help="Archivos tar o csv de entrada")
    parser.add_argument("-c", "--cleanup", action="store_true", help="Realizar limpieza de la unión")
    parser.add_argument("-n", "--name", type=str, default="csv/Merged_DF.csv", help="Nombre del csv final")
    args = parser.parse_args()
    
    csvs = []
    for name in args.input:
        try:
            if name.endswith('.tar.xz'):
                print(f'Descomprimiendo {name}...')
                
                with tarfile.open(name, "r:xz") as tar:
                    member = next(m for m in tar.getmembers() if m.name.endswith('.csv'))
                    f = tar.extractfile(member)
                    df = pd.read_csv(io.BytesIO(f.read()))
            else:
                df = pd.read_csv(name)
        except Exception as e:
            print(f"Error al procesar {name}: {e}")
            return

        
        print(f"Cargando archivo: {name} || Columnas: {len(df.columns)}")
            
        csvs.append(df)
        
    if len(csvs) > 1:
        merged = mergecsvs(csvs)
    else:
        merged = csvs.pop()
    if args.cleanup == True:
        print("DataFrame unido, limpiando...")
        merged = protocol_extractor(merged)
        merged = attack_bening_basic_cleanup(merged)
    os.makedirs("csv", exist_ok=True)
    
    if args.name.endswith(".csv") == False:
        args.name = f"{args.name}.csv"
    args.name = f"{args.name}"
    print(f"Dataframe nuevo, guardando como {args.name}...")
    merged.to_csv(args.name, index=False)
    
if __name__ == "__main__":
    main()
    
    