import tarfile
import pandas as pd
import io
import os
import argparse
import ast
from sklearn.preprocessing import LabelEncoder
from enum import Enum
import joblib

class ClassIdentifier(Enum):
    BINARIO = "1"
    CLASS8 = "2"
    CLASS61 = "3"
    NONE = "0"

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
    
    #Quitar columnas con datos iguales
    cols = [col for col in df.columns if df[col].nunique() <= 1]
    print(f"Se han borrado {len(cols)} columnas con los mismos valores del DataFrame: \n{cols}")
    df = df.drop(cols, axis=1)
    df = df.select_dtypes(include=['number'])
    return df

def class_classifier(df: pd.DataFrame, numcl: ClassIdentifier) -> pd.DataFrame:
    le = LabelEncoder()
    value = numcl.value
    if value == ClassIdentifier.BINARIO.value:
        df['target'] = df['label1'].map({'benign': 0, 'attack': 1})
    elif value == ClassIdentifier.CLASS8.value:
        df['target'] = le.fit_transform(df['label2'])
        joblib.dump(le, 'joblibs/le_class.joblib')
    else:
        df['target'] = le.fit_transform(df['label3'])
        joblib.dump(le, 'joblibs/le_class.joblib')
        
    if value != ClassIdentifier.BINARIO.value:
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        print(f"Diccionario de etiquetas: {mapping}")
        
    df = df.drop(['device_name', 'device_mac', 'label_full', 'label1', 'label2', 'label3', 'label4', 'timestamp'], axis=1)
    return df

def class_menu():
    print("\nSelecciona las clases a diferenciar:")
    print("1. Clasificación Binaria (Ataque/Benigno)")
    print("2. Clasificación Multiclase Básica (Tipo de ataque)")
    print("3. Clasificación Multiclase Profunda (Tipo de ataque)")
    while True:
        opcion = input("Elige una opcion: ")
        if opcion == "1":
            return ClassIdentifier.BINARIO
        if opcion == "2":
            return ClassIdentifier.CLASS8
        if opcion == "3":
            return ClassIdentifier.CLASS61
        else:
            print("Opcion no valida")
            
def feature_selection_params(df: pd.DataFrame) -> pd.DataFrame:
    return df[['log_messages_count', 'log_data-types', 'network_fragmented-packets', 'network_ip-flags_max', 'network_tcp-flags-psh_count']]
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", help="Archivos tar o csv de entrada")
    parser.add_argument("-c", "--cleanup", action="store_true", help="Realizar limpieza de la unión")
    parser.add_argument("-cF", "--cleanupFeature", action="store_true", help="Realizar limpieza del Dataframe siguiendo el Feature selection")
    parser.add_argument("-n", "--name", type=str, default="csv/Merged_DF.csv", help="Nombre del csv final")
    parser.add_argument("-C", "--classes", type=str, choices=[e.value for e in ClassIdentifier], default=ClassIdentifier.NONE.value, help="Número de clases a diferenciar: 1: binario, 2: 8 clases, 3: 61 clases")
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
        
        if args.classes == ClassIdentifier.NONE.value:
            args.classes = class_menu()
        
        merged = class_classifier(merged,args.classes)
        merged = attack_bening_basic_cleanup(merged)
    elif args.cleanupFeature == True:
        if args.classes == ClassIdentifier.NONE.value:
            args.classes = class_menu()
            
        merged = feature_selection_params(merged)
        
    os.makedirs("csv", exist_ok=True)
    
    if args.name.endswith(".csv") == False:
        args.name = f"{args.name}.csv"
    args.name = f"{args.name}"
    print(f"Dataframe nuevo, guardando como {args.name}...")
    merged.to_csv(args.name, index=False)
    
if __name__ == "__main__":
    main()
    
    