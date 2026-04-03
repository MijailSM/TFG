import dask.dataframe as dd
from dask.distributed import Client

def process():
    client = Client()
    print(f"Dashboard en {client.dashboard_link}")

    # 1. Carga virtual del archivo (no ocupa RAM al inicio)
    df = dd.read_csv('/mnt/datos/AllCicMerged.csv')

    # 2. Contar etiquetas de forma masiva usando todos los núcleos
    counts = df['label1'].value_counts().compute() 

    print(counts)
    
if __name__ == "__main__":
    process()