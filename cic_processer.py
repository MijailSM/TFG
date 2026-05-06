import dask.dataframe as dd
from dask.distributed import Client
import os

def balance_data():
    # 1. Configuración de recursos muy conservadora
    # 1 thread por worker evita que compitan por la misma RAM
    with Client(n_workers=4, threads_per_worker=1, memory_limit='2GB') as client:
        
        # Leer con blocksize pequeño para no saturar
        df = dd.read_csv('/mnt/datos/AllCicMerged.csv', blocksize="100MB")

        # 2. Obtener conteos (esto es rápido)
        print("Obteniendo conteos...")
        counts = df['label2'].value_counts().compute()
        counts_dict = counts.to_dict()  # ← acceso por dict, no por índice de Dask

        print("Clases encontradas:", list(counts_dict.keys()))

        if 'benign' not in counts_dict:
            raise ValueError(f"'benign' no encontrado. Clases: {list(counts_dict.keys())}")

        target_count = counts_dict['benign']
        
        # 3. Lista para guardar los fragmentos procesados
        sampled_fragments = []

        print("Iniciando muestreo por clase (evitando groupby)...")
        for label, count in counts.items():
            # Filtramos la clase actual
            condition = (df['label2'] == label)
            class_df = df[condition]
            
            clean_df = class_df[class_df['Flow Duration'] != 0]
            clean_count = clean_df.shape[0].compute()

            if clean_count == 0:
                print(f" - AVISO: {label} no tiene filas válidas (Duration > 0), saltando.")
                continue
            
            if count > target_count:
                # Si es mayor que benign, calculamos fracción para reducir
                frac = target_count / count
                print(f" - Reduciendo {label} al {frac*100:.2f}%")
                sampled_fragments.append(class_df.sample(frac=frac, random_state=42))
            else:
                # Si es menor o igual, la dejamos completa
                print(f" - Manteniendo {label} al 100%")
                sampled_fragments.append(class_df)

        # 4. Concatenar todos los fragmentos muestreados
        balanced_df = dd.concat(sampled_fragments)

        # 5. Guardar (IMPORTANTE: Usar un directorio, no un archivo único)
        output_dir = '/mnt/datos/output/balanced_csv_parts/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Guardando particiones en {output_dir}...")
        # El '*' es vital para que Dask escriba en paralelo sin colapsar
        balanced_df.to_csv(output_dir + 'part-*.csv', index=False)
        
        print("¡Hecho!")

if __name__ == "__main__":
    balance_data()