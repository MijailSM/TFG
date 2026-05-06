import tqdm
import pandas as pd
import glob

files = glob.glob('/mnt/datos/output/balanced_csv_parts/*.csv')
archivo_salida = 'mnt/datos/output/MergedBalanced.csv'
first = True
for file in tqdm.tqdm(files, desc="Procesando csvs...", unit='file'):
    try:
        df = pd.read_csv(file)
        
        if first:
            df.to_csv(archivo_salida, index=False, mode='w', encoding='utf-8')
            first = False
        else:
            df.to_csv(archivo_salida, index=False, mode='a', encoding='utf-8', header=False)
        del df
    except Exception as e:
        print(f"ERROR en {file}: {e}")