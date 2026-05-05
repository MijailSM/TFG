import dask.dataframe as dd
from dask.distributed import Client
import dask
import os

def balance_data():
    tmp_dir = "/mnt/datos/dask-temp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Iniciamos el cliente para ver el dashboard y mejorar el manejo de memoria
    with Client(n_workers=4, threads_per_worker=2, memory_limit='4GB') as client:
        print(f"Dashboard en {client.dashboard_link}")

        # 1. Cargar el dataset
        df = dd.read_csv('/mnt/datos/AllCicMerged.csv')

        # 2. Obtener los conteos actuales
        print("Calculando frecuencias originales...")
        counts = df['label2'].value_counts().compute()
        
        target_count = counts['benign']
        print(f"Valor de referencia (benign): {target_count}")

        # 3. Calcular fracciones de muestreo dinámicas
        # Si count > target: fraccion = target / count (reducimos)
        # Si count <= target: fraccion = 1.0 (mantenemos todo)
        fractions = {
            label: min(1.0, target_count / count) 
            for label, count in counts.items()
        }

        print("Fracciones calculadas para el balanceo:")
        for label, f in fractions.items():
            print(f" - {label}: {f:.4f}")

        # 4. Aplicar el muestreo
        # Usamos groupby sobre 'label2' para aplicar la fracción correspondiente a cada grupo
        balanced_df = df.groupby('label2').apply(
            lambda x: x.sample(frac=fractions[x.name], random_state=42), 
            meta=df._meta
        )

        # 5. Guardar el resultado
        # Nota: to_csv generará múltiples archivos (particiones). 
        # Si quieres un solo archivo usa single_file=True (cuidado con el tamaño de RAM)
        output_path = '/mnt/datos/AllCicMerged_Balanced_Selective.csv'
        print(f"Guardando datos balanceados en {output_path}...")
        
        balanced_df.to_csv(output_path, index=False, single_file=True)
        
        print("Muestreo completado y guardado exitosamente.")

if __name__ == "__main__":
    with dask.config.set({'temporary_directory': '/mnt/dask-temp'}):
        balance_data()