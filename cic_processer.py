import dask.dataframe as dd
from dask.distributed import Client
import dask
import os

def process():
    with Client() as client:
        print(f"Dashboard en {client.dashboard_link}")

        df = dd.read_csv('/mnt/datos/AllCicMerged.csv')

        counts = df['label2'].value_counts().compute() 

        print("\n--- RESULTADOS ---")
        print(counts)
        print("------------------\n")
        
import dask.dataframe as dd
from dask.distributed import Client

def balance_data():
    tmp_dir = "/mnt/datos/dask-temp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    with Client() as client:
        df = dd.read_csv('/mnt/datos/AllCicMerged.csv')

        # 1. Obtener el conteo de la clase minoritaria
        # Esto dispara un cómputo pequeño pero necesario
        counts = df['label1'].value_counts().compute()
        min_class_count = counts.min()
        
        # 2. Calcular las fracciones de muestreo para cada clase
        # Si la clase A tiene 1000 y la minoritaria es 100, la fracción es 0.1
        fractions = {label: min_class_count / count for label, count in counts.items()}

        # 3. Aplicar el muestreo usando una función personalizada por partición
        # .sample() en Dask permite pasar un diccionario de fracciones si se usa con groupby
        balanced_df = df.groupby('label1').apply(
            lambda x: x.sample(frac=fractions[x.name]), 
            meta=df._meta
        )

        # 4. (Opcional) Guardar el resultado balanceado
        # balanced_df sigue siendo "lazy", no se ha procesado hasta aquí
        balanced_df.to_csv('/mnt/datos/AllCicMerged_Balanced-Binary.csv')
        
        print("Muestreo completado y guardado.")

if __name__ == "__main__":
    with dask.config.set({'temporary_directory': '/mnt/datos/temp'}):
        balance_data()