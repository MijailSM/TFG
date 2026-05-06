import dask.dataframe as dd
from dask.distributed import Client
import numpy as np
import os

def balance_data():
    # 1. Configuración de recursos
    with Client(n_workers=4, threads_per_worker=1, memory_limit='2GB') as client:
        
        # Forzamos tipos para evitar errores de inferencia
        tipos = {
            'label2': 'string',
            'Flow Duration': 'float64'
        }

        print("Leyendo dataset...")
        df = dd.read_csv('/mnt/datos/AllCicMerged.csv', blocksize="100MB", dtype=tipos)

        # 2. Definir el objetivo (basado en 'benign')
        # Calculamos el conteo inicial solo para saber cuánto necesitamos
        counts = df['label2'].value_counts().compute()
        target_count = counts['benign']
        
        sampled_fragments = []

        print(f"Iniciando limpieza y balanceo. Objetivo: {target_count} filas reales por clase...")

        for label, count in counts.items():
            # A. Filtramos la clase actual
            class_df = df[df['label2'] == label]
            
            # B. LIMPIEZA RADICAL: Solo nos quedamos con lo que sirve
            # Reemplazamos infinitos por NaN para poder usar dropna de forma uniforme
            class_df = class_df.replace([np.inf, -np.inf], np.nan)
            
            # Eliminamos filas donde 'Flow Duration' sea NaN o 0
            # Esto soluciona el error de la imagen al limpiar la columna antes del filtro lógico
            clean_df = class_df.dropna(subset=['Flow Duration'])
            clean_df = clean_df[clean_df['Flow Duration'] > 0]
            
            # C. Conteo de filas "reales" disponibles
            available_clean_count = len(clean_df)
            print(f" - Clase {label}: {available_clean_count} filas válidas encontradas.")

            if available_clean_count == 0:
                print(f" - AVISO: {label} no tiene datos válidos. Saltando.")
                continue

            # D. Muestreo sobre los datos limpios
            if available_clean_count > target_count:
                # Si sobran, reducimos para igualar a benign
                frac = target_count / available_clean_count
                sampled_fragments.append(clean_df.sample(frac=frac, random_state=42))
                print(f"   -> Muestreadas {target_count} filas (Reducción).")
            else:
                # Si faltan o es igual, nos quedamos con todos los que sean "reales"
                sampled_fragments.append(clean_df)
                print(f"   -> Manteniendo todas las {available_clean_count} filas válidas.")

        # 3. Consolidación y Guardado
        print("Concatenando resultados limpios...")
        balanced_df = dd.concat(sampled_fragments)

        output_dir = '/mnt/datos/output/balanced_real_data/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Guardando en {output_dir}...")
        balanced_df.to_csv(output_dir + 'part-*.csv', index=False)
        
        print("¡Hecho! El CSV resultante ya no tiene infinitos ni duraciones nulas.")

if __name__ == "__main__":
    balance_data()