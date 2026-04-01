import pandas as pd
import re
import os
import glob
import tqdm

def csv_putLabel(df: pd.DataFrame, name: str):
    labels = re.findall(r'([^_.])', name.split('.')[0])
    if 'benign' in labels:
        for label in labels:
            label = 'benign'
    
    for i in range(3):
        df[f'label{i+1}'] = labels[i]
    
    return df

root_dir = '/mnt/datos/csvs/'
archivo_salida = '/mnt/datos/AllCicMerged.csv'

files = glob.glob(os.path.join(root_dir, '**/*.csv'), recursive=True)

first = True

for file in tqdm.tqdm(files, desc="Procesando csvs...", unit='file'):
    try:
        df = pd.read_csv(file)
        
        df = csv_putLabel(df, file)
        
        if first:
            df.to_csv(archivo_salida, index=False, mode='w', encoding='utf-8')
        else:
            df.to_csv(archivo_salida, index=False, mode='a', encoding='utf-8')
        del df
    except Exception as e:
        print(f"ERROR en {file}: {e}")