import pandas as pd
import tqdm

file = '/mnt/datos/AllCicMerged.csv'
label = 'label1'

conteos = {}

reader = pd.read_csv(file, usecols=[label], chunksize=10000)

for chunk in tqdm.tqdm(reader, desc="Leyendo etiqueta...", unit="Chunk"):
    conteo = chunk[label].value_counts()
    
    for value, count in conteo.items():
        conteos[label] = conteos.get(label, 0) + value
        
total = sum(conteos.values())
print("RESUMEN")
for value, count in conteos.items():
    percentage = (count / total) * 100
    print(f"{value}: {count} filas ({percentage:.2f}%)")