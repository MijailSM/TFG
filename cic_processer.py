import dask.dataframe as dd

# 1. Carga virtual del archivo (no ocupa RAM al inicio)
df = dd.read_csv('/mnt/datos/AllCicMerged.csv')

# 2. Contar etiquetas de forma masiva usando todos los núcleos
counts = df['label1'].value_counts().compute() 

print(counts)