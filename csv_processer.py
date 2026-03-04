import tarfile
import pandas as pd
import io
import os
import argparse
import ast
from sklearn.preprocessing import LabelEncoder
from enum import Enum
import joblib
from collections import Counter

class ClassIdentifier(Enum):
    BINARIO = "bin"
    CLASS8 = "C8"
    CLASS60 = "C60"

def mergecsvs(csvs: list) -> pd.DataFrame:
    
    print("Uniendo todo...")
    merged = pd.concat(csvs, ignore_index=True)
    
    merged = merged.sample(frac=1).reset_index(drop=True)
    
    print(f"porcentaje de cada clase:\n{merged['label1'].value_counts(normalize=True) * 100}")
    
    return merged

def protocol_extractor(df: pd.DataFrame) -> pd.DataFrame:
    #Usando one hot encoding
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

def data_type_extractor(df: pd.DataFrame) -> pd.DataFrame:
    df["data_types_list"] = df["log_data-types"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
    )
    data_types = df["data_types_list"].explode().unique()
    for tipo in data_types:
        df[f"types_data_{tipo}"] = df["data_types_list"].apply(
            lambda x: 1 if tipo in x else 0
        )
    df = df.drop(columns=["data_types_list", "log_data-types"])
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

    columns_to_drop = ['device_name', 'device_mac', 'label_full', 'label1', 'label2', 'label3', 'label4', 'timestamp']
    columns_existentes = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(columns_existentes, axis=1)
    return df
            
def allportExtractor(df: pd.DataFrame) -> pd.DataFrame:
    #Usando Frequency Encoding
    df["ports_list"] = df['network_ports_all'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    allports = [p for sublista in df["ports_list"] for p in sublista]
    frecuencias = Counter(allports)
    def avg_mean(port_list):
        if not port_list: return 0
        return sum(frecuencias[p] for p in port_list) / len(port_list)
    df["port_frequency_avg"] = df["ports_list"].apply(avg_mean)
    return df
            
def feature_selection_params(df: pd.DataFrame, whitelist: list = None) -> pd.DataFrame:
    
    df = allportExtractor(df)
    df = data_type_extractor(df)
    prefix = 'types_data_'
    columnas_data_type = [col for col in df.columns if col.startswith(prefix)]
    return df[[
        #FALTA ALL IPs
        'log_messages_count', 
        'log_data-ranges_avg',
        'network_fragmented-packets', 
        'network_interval-packets',
        'network_packets_all_count',
        'network_ips_all_count',
        'network_packet-size_std_deviation',
        'network_protocols_all_count',
        'network_time-delta_avg',
        'network_ttl_avg',
        'network_window-size_avg',
        'network_ip-flags_max', 
        'network_tcp-flags-psh_count',
        'port_frequency_avg',
    ] + (whitelist if whitelist else []) + (columnas_data_type)]
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", help="Archivos tar o csv de entrada")
    parser.add_argument("-c", "--cleanup", action="store_true", help="Realizar limpieza de la unión")
    parser.add_argument("-cF", "--cleanupFeature", action="store_true", help="Realizar limpieza del Dataframe siguiendo el Feature selection")
    parser.add_argument("-n", "--name", type=str, default="csv/Merged_DF.csv", help="Nombre del csv final")
    parser.add_argument("-C", "--classes", action="store_true", help="Exportar 3 clases distintas")
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
        
        merged = class_classifier(merged,args.classes)
        merged = attack_bening_basic_cleanup(merged)
    elif args.cleanupFeature == True:
        print("Usando feature_selection...")
        whitelist = ['label_full', 'label1', 'label2', 'label3', 'label4']
        merged = feature_selection_params(merged, whitelist)
        
    #os.makedirs("csv", exist_ok=True)
    
    if args.name.endswith(".csv") == False:
        args.name = f"{args.name}.csv"
    args.name = f"{args.name}"
    
    if args.classes == True:
        for c in ClassIdentifier:
            fdf = class_classifier(merged, c)
            print(f"Dataframe nuevo, guardando como {c.value}{args.name}...")
            fdf.to_csv(f"{c.value}{args.name}", index=False)
    else:    
        print(f"Dataframe nuevo, guardando como {args.name}...")
        merged.to_csv(args.name, index=False)
    
if __name__ == "__main__":
    main()
    
    