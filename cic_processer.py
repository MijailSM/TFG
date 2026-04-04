import dask.dataframe as dd
from dask.distributed import Client

def process():
    with Client() as client:
        print(f"Dashboard en {client.dashboard_link}")

        df = dd.read_csv('/mnt/datos/AllCicMerged.csv')

        counts = df['label2'].value_counts().compute() 

        print("\n--- RESULTADOS ---")
        print(counts)
        print("------------------\n")

if __name__ == "__main__":
    process()