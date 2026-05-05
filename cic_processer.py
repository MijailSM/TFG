import dask.dataframe as dd
from dask.distributed import Client
import dask
import os

def balance_data():
    tmp_dir = "/mnt/datos/dask-temp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Configuramos el cliente con límites más estrictos para evitar el crash de memoria
    with Client(n_workers=4, threads_per_worker=1, memory_limit='2GB') as client:
        print(f"Dashboard en {client.dashboard_link}")

        # 1. Cargar el dataset con blocksize pequeño para controlar RAM
        df = dd.read_csv('/mnt/datos/AllCicMerged.csv', blocksize="100MB")

        print("Calculando frecuencias originales...")
        counts = df['label2'].value_counts().compute()
        
        target_count = counts['benign']
        print(f"Valor de referencia (benign): {target_count}")

        fractions = {
            label: min(1.0, target_count / count) 
            for label, count in counts.items()
        }

        # 4. Aplicar el muestreo
        # El truco aquí es usar group_keys=False si fuera pandas, 
        # pero en Dask lo resolvemos reseteando el índice después.
        balanced_df = df.groupby('label2', group_keys=False).apply(
            lambda x: x.sample(frac=fractions[x.name], random_state=42), 
            meta=df.dtypes.to_dict() # Usamos los tipos directos como meta
        )

        # 5. IMPORTANTE: Limpiar el índice
        # groupby.apply suele crear un índice multinivel que to_csv no maneja bien
        balanced_df = balanced_df.reset_index(drop=True)

        output_path = '/mnt/datos/output/AllCicMerged_Balanced_Selective-*.csv'
        print(f"Guardando datos balanceados en {output_path}...")
        
        # Guardamos como múltiples archivos para no saturar la RAM
        balanced_df.to_csv(output_path, index=False)
        
        print("Muestreo completado y guardado exitosamente.")

if __name__ == "__main__":
    # Asegúrate de que esta ruta tenga permisos de escritura
    with dask.config.set({'temporary_directory': '/mnt/datos/dask-temp'}):
        balance_data()