import pandas as pd

# Contar cabeceras repetidas sin cargar el fichero entero
chunk_iter = pd.read_csv('/mnt/datos/AllCicMerged.csv', chunksize=100_000)
header_rows = 0
label2_values = {}

for chunk in chunk_iter:
    # Cabeceras repetidas aparecen como filas donde label2 == 'label2'
    header_rows += (chunk['label2'] == 'label2').sum()
    for v, c in chunk['label2'].value_counts().items():
        label2_values[v] = label2_values.get(v, 0) + c

print("Filas con cabecera repetida:", header_rows)
print("Clases encontradas:", label2_values)