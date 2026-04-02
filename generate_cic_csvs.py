import pandas as pd
import re
import os
import glob
import tqdm

def csv_putLabel(df: pd.DataFrame, file_path: str):
    name = os.path.basename(file_path).split('.')[0]
    labels = re.split(r'[_.]', name)
    
    if 'benign' in labels:
        labels = ['benign'] * 3 
    while len(labels) < 3:
        labels.append('none')

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
            first = False
        else:
            df.to_csv(archivo_salida, index=False, mode='a', encoding='utf-8', header=False)
        del df
    except Exception as e:
        print(f"ERROR en {file}: {e}")