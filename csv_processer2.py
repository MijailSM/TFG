import pandas as pd
import logging
import tqdm
import tarfile
import io
import argparse
import ast
from group_lasso import GroupLasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pygad
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import RandomUnderSampler
import collections

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

custom_bar = "{l_bar}{bar:20}{r_bar}{bar:-20b}"

context_groups = {
    'Identifiers': 1,      # IPs, IDs, Timestamp
    'Time_Metrics': 2,     # Duration, IAT, Idle, Active
    'Packet_Size': 3,      # Length, Size, Segment, Bytes
    'Header_Flags': 4,     # TCP Flags, Header_Length
    'Port_Service': 5,     # Source/Dest Ports, Freq_Port
    'Flow_Traffic': 6,     # Packets/s, Bytes/s, Flow_Packets
    'Subflow_Stats': 7,    # Subflow Fwd/Bwd
    'Window_Dynamics': 8,  # Window size, Init_Win
    'Payload_Stats': 9,    # Bulk, Down/Up Ratio, Payload
    'Protocol_Ind': 10     # Protocol, OHE_Protocol
}

mapping = {}


def mergecsvs(csvs: list) -> pd.DataFrame:
    """Merges two or more csvs in one Dataframe

    Args:
        csvs (list): Dataframes to merge

    Returns:
        pd.DataFrame: merged Dataframe
    """
    
    logging.info("Uniendo todo...")
    merged = pd.concat(csvs, ignore_index=True)
    
    merged = merged.sample(frac=1).reset_index(drop=True)
    
    logging.info(f"porcentaje de cada clase:\n{merged['label1'].value_counts(normalize=True) * 100}")
    
    return merged

def tar_extract(filename: str):
    """Extracts a Dataframe in a tar.xz type compression

    Args:
        filename (str): filename to extract

    Returns:
        pd.DataFrame: bytes decompressed
    """
    
    logging.info(f"Extrayendo {filename}...")
    try:
        with tarfile.open(filename, "r:xz") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith('.csv'))
            f = tar.extractfile(member)
            df = pd.read_csv(io.BytesIO(f.read()))
    except Exception as e:
        logging.exception(e)
        exit(1)
    return df

def undersampling(df: pd.DataFrame, target, samples=15000) -> pd.DataFrame:
    # counts = df["target"].value_counts()
    # df = df[df[target].isin(counts[counts > 10].index)]
    
    counter = collections.Counter(target)
    strategy = {
        label: min(count, samples) for label, count in counter.items()
    }
    if len(counter) > 2:
        rus = RandomUnderSampler(random_state=42, sampling_strategy=strategy)
    else:
        rus = RandomUnderSampler(random_state=42)
    X_res, target = rus.fit_resample(df, target)
    df_res = pd.DataFrame(X_res, columns=df.columns)
    return df_res, target

def limpieza_basica(df: pd.DataFrame) -> pd.DataFrame:
    #TODO
    pass

def feature_selection_CICcols(df: pd.DataFrame, whitelist: list=None) -> pd.DataFrame:
    df = clean_csv_lists(df)
    
    df, newfcols = frequency_encoding(df, 'network_ports_all')
    df, newohecols = frequency_encoding(df, 'log_data-types')
    
    fixed_cols = [
        'log_messages_count', 'log_data-ranges_avg',
        'network_fragmented-packets', 'network_interval-packets',
        'network_packets_all_count', 'network_ips_all_count',
        'network_packet-size_std_deviation', 'network_protocols_all_count',
        'network_time-delta_avg', 'network_ttl_avg',
        'network_window-size_avg', 'network_ip-flags_max', 
        'network_tcp-flags-psh_count'
    ]
    
    whitelist = whitelist if whitelist else []
    
    all_final_cols = fixed_cols + whitelist + newohecols + newfcols
    
    existing_cols = [c for c in all_final_cols if c in df.columns]
    
    return df[existing_cols]

def first_clean(df: pd.DataFrame, classes: int):
    
    if classes == 1:
        target = 'label1'
    elif classes == 2:
        target = 'label2'
    elif classes == 3:
        target = 'label3'
    
    df = df.drop(columns=[
        'timestamp', 
        'timestamp_start', 
        'timestamp_end', 
        'network_ips_all', 
        'network_ips_dst',
        'network_ips_src',
        'network_macs_all',
        'network_macs_dst',
        'network_macs_src',
        'device_name',
        'device_mac'
    ])
    
    y = df[target]
    
    le = LabelEncoder()
    y = le.fit_transform(df[target])
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    
    df = df.drop(columns=['label_full', 'label1', 'label2', 'label3', 'label4'])
    return df, y

    
def clean_csv_lists(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.select_dtypes(include=['object', 'str']).columns
    for col in tqdm.tqdm(cols, desc="Limpiando listas...", unit="columna", bar_format=custom_bar):
        sample_val = str(df[col].iloc[0])
        if sample_val.startswith('[') and sample_val.endswith(']'):
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def frequency_encoding(df: pd.DataFrame, col) -> tuple[pd.DataFrame, list]:
    is_list = isinstance(df[col].iloc[0], list)
    if is_list:
        flat = [item for sublist in df[col].dropna() for item in sublist]
        counts = pd.Series(flat).value_counts(normalize=True).to_dict()
        new_name = f"freq_{col}"
        df[new_name] = df[col].apply(lambda x: sum(counts.get(i, 0) for i in x)/len(x) if isinstance(x, list) and len(x)>0 else 0)
    else:
        counts = df[col].value_counts(normalize=True).to_dict()
        new_name = f"freq_{col}"
        df[new_name] = df[col].map(counts)
    df.drop(columns=[col], inplace=True)
    
    return df, [new_name]

def one_hot_encoding(df: pd.DataFrame, col) -> tuple[pd.DataFrame, list]:
    dummies = pd.get_dummies(df[col], prefix=f"OHE_{col}")
    new_cols = dummies.columns.to_list()
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=[col], inplace=True)
    return df, new_cols
    

def feature_selection(df: pd.DataFrame, classes: int, threshold: int=15) -> pd.DataFrame:
    
    df, y = first_clean(df, classes)
    
    cols = df.select_dtypes(include=['str', 'object']).columns
    df = clean_csv_lists(df)
    feature_groups = {c: c for c in df.select_dtypes(exclude=['object', 'str']).columns}
    
    #STAGE 0: CLEANING UP
    for col in tqdm.tqdm(cols, desc="Encodeando columnas...", unit="columna", bar_format=custom_bar):
        is_list = isinstance(df[col].iloc[0], list)
        
        if is_list:
            all_elements = [item for sublist in df[col] for item in sublist]
            n_unique = len(set(all_elements))
        else:
            n_unique = df[col].nunique()
        
        if n_unique > threshold or is_list:
            df, created_cols = frequency_encoding(df, col)
        else:
            df, created_cols = one_hot_encoding(df, col)
            
        for c in created_cols:
            feature_groups[c] = col
            
    #STAGE 1 CONTEXT GROUPING
    gid = []
    
    for col in tqdm.tqdm(df.columns, desc="Agrupando en columnas", bar_format=custom_bar):
        orig = feature_groups.get(col, col).lower()
        
        if any(x in orig for x in ['device', 'timestamp']):
            g_id = 1
        elif 'interval' in orig or 'time-delta' in orig:
            g_id = 2
        elif 'log_data' in orig or 'log_messages' in orig:
            g_id = 3
        elif 'header-length' in orig or 'ip-flags' in orig:
            g_id = 4
        elif 'packet-size' in orig or 'ip-length' in orig:
            g_id = 5
        elif 'payload-length' in orig or 'mss' in orig:
            g_id = 6
        elif 'ips' in orig or 'macs' in orig:
            g_id = 7
        elif 'ports' in orig:
            g_id = 8
        elif 'count' in orig or 'fragmentation' in orig:
            g_id = 9
        elif 'tcp-flags' in orig or 'window-size' in orig or 'ttl' in orig:
            g_id = 10
        else:
            g_id = 9 # Por defecto a estadísticas de flujo
            
        gid.append(g_id)
        
    #STAGE 2 SPARSE-GROUP LASSO
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    groups = np.array(gid)
    
    sgl = GroupLasso(
        groups=groups,
        group_reg=0.01, #Penalización por bloque
        l1_reg=0.05, #Penalizacion por variable
        fit_intercept=True,
        n_iter=500, 
        random_state=42
    )
    
    logging.info("Ejecutando Sparse Group Lasso...")
    sgl.fit(X_scaled, y)
    
    mask = np.abs(sgl.coef_).flatten() > 1e-5
    
    selected_cols = df.columns[mask].to_list()
    new_gid = np.array(gid)[mask].tolist()
    
    #STAGE 3 RRS TABLES
    df_pruned = df[selected_cols]
    logging.info(f"Calculando RRS Scores para {len(selected_cols)} columnas...")
    
    mi_scores = mutual_info_classif(df_pruned, y, random_state=42)
    rrs_table = pd.Series(mi_scores, index=selected_cols).sort_values(ascending=False)
    
    #STAGE 4 Algoritmo genético
    num_features = df_pruned.shape[1]
    features_names = df_pruned.columns.tolist()
    
    def fitness_fun(ga_instance, solution, solution_idx):
        idx_selected = [i for i, bit in enumerate(solution) if bit == 1]
        
        if len(idx_selected) == 0:
            return 0
        
        X_sub = df_pruned.iloc[:, idx_selected]
        
        clf = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=-1, random_state=42)
        scores = cross_val_score(clf, X_sub, y, cv=3)
        
        return scores.mean()
    
    # El paper dice: Probabilidad de bit 1 = Score RRS normalizado
    normalized_rrs = (rrs_table - rrs_table.min()) / (rrs_table.max() - rrs_table.min())
    initial_population = []
    
    for _ in range(50):
        chromosome = [1 if np.random.rand() < normalized_rrs[feat] else 0 for feat in features_names]
        initial_population.append(chromosome)
    
    ga_instance = pygad.GA(
        num_generations=20,
        num_parents_mating=10,
        fitness_func=fitness_fun,
        initial_population=initial_population,
        # En lugar de porcentaje, definimos el número exacto de genes a mutar
        mutation_num_genes=1, 
        mutation_type="random",
        crossover_type="single_point",
        parent_selection_type="sss", 
        keep_parents=2,
        suppress_warnings=True # Esto silenciará el aviso automáticamente
    )
    
    logging.info("Iniciando algoritmo genetico...")
    ga_instance.run()
    
    solution, solution_fitness, _ = ga_instance.best_solution()
    final_features = [features_names[i] for i, bit in enumerate(solution) if bit == 1]
            
    return df[final_features], final_features
    
    
            
    
            
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", help="Archivos tar o csv de entrada")
    parser.add_argument("-b", "--basica", action="store_true", help="Realizar limpieza básica de los csvs", default=False)
    parser.add_argument("-Fd", "--feature_selection_columns", action="store_true", help="Solo conserva las columnas del feature selection del CIC", default=False)
    parser.add_argument("-Fs", "--feature_selection", action="store_true", help="Se realiza el algoritmo de Feature Selection", default=False)
    parser.add_argument("-T", "--target_list", action="store_true", help="Mostrar todos los target", default=False)
    parser.add_argument("-n", "--name", type=str, default="Merged_DF.csv", help="Nombre del csv final")
    
    #Añadir las columnas una vez terminado el feature selection
    args = parser.parse_args()
    
    csvs = []
    for filename in tqdm.tqdm(args.input, desc="Procesando archivos...", unit="archivo", bar_format=custom_bar):
        if filename.endswith(".tar.xz"):
            f = tar_extract(filename)
            df = pd.read_csv(io.BytesIO(f.read()))
        else:
            df = pd.read_csv(filename)
        csvs.append(df)
        
    if len(csvs) > 1:
        dfmerged = mergecsvs(csvs)
    else:
        dfmerged = csvs.pop()
        
    if args.basica == True:
        dfmerged = limpieza_basica(dfmerged)
    elif args.feature_selection_columns == True:
        whitelist = ['label_full', 'label1', 'label2', 'label3', 'label4']
        dfmerged = feature_selection_CICcols(dfmerged, whitelist)
    elif args.feature_selection == True:
        dfmerged, final_features = feature_selection(dfmerged, 1)
        logging.info(f"Las columnas generadas son {final_features}")
    
    if args.feature_selection == False:
        for i in tqdm.tqdm(range(0, 3), desc="Extrayendo csvs...", unit="archivo", bar_format=custom_bar):
            if i == 0:
                target = 'label1'
            elif i == 1:
                target = 'label2'
            else:
                target = 'label3'
            
            le = LabelEncoder()
            y = le.fit_transform(dfmerged[target])
            target_cols = ['label_full', 'label1', 'label2', 'label3', 'label4']
            df_aux = dfmerged.drop(columns=target_cols)
            #df_aux, y = undersampling(df_aux, y)
            df_aux['target'] = y
            
            if args.name.endswith('.csv') == False:
                args.name = args.name + '.csv'
            logging.info(f"Guardando csv como: {target}_{args.name}...")
            df_aux.to_csv(f"csv/{target}_{args.name}", index=False)
        
        
        
        
