import dask.dataframe as dd

def procesar_etiquetas(path_entrada, path_salida):
    # 1. Cargar el dataset (ajusta el separador si es CSV, TSV, etc.)
    # Dask no carga los datos en memoria hasta que se solicita un cálculo
    df = dd.read_csv(path_entrada)

    # 2. Filtrar o seleccionar las columnas de interés
    columnas_objetivo = ['label1', 'label2', 'label3']
    
    # 3. Realizar el conteo de valores
    # .value_counts() en Dask devuelve una Serie con los conteos
    # Usamos .compute() para ejecutar el cálculo paralelo y obtener el resultado
    print("Calculando frecuencias...")
    
    resultados = {}
    for col in columnas_objetivo:
        if col in df.columns:
            resultados[col] = df[col].value_counts().compute()
    
    # 4. Escribir los resultados en un archivo de texto
    with open(path_salida, 'w', encoding='utf-8') as f:
        f.write("REPORTE DE CONTEO DE ETIQUETAS\n")
        f.write("==============================\n\n")
        
        for col, counts in resultados.items():
            f.write(f"--- Columna: {col} ---\n")
            f.write(counts.to_string())
            f.write("\n\n")
            
    print(f"Proceso finalizado. Resultados guardados en: {path_salida}")

# Uso del script
if __name__ == "__main__":
    archivo_datos = "/mnt/datos/AllCicMerged.csv"  # Cambia por tu archivo
    archivo_reporte = "conteo_labels.txt"
    
    procesar_etiquetas(archivo_datos, archivo_reporte)