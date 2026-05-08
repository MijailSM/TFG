import dask.dataframe as dd
from dask.distributed import Client
import numpy as np
import os

COLS_WITH_INF = ['Flow Bytes/s', 'Flow Packets/s']

def limpiar_particion(df):
    """Elimina filas con Inf o NaN en las columnas problemáticas de CICFlowmeter."""
    df = df.copy()
    df[COLS_WITH_INF] = df[COLS_WITH_INF].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=COLS_WITH_INF)
    df = df[df['Flow Duration'] > 0]
    return df

def balance_data():
    with Client(n_workers=4, threads_per_worker=1, memory_limit='2GB') as client:

        tipos = {
            'label2': 'string',
            'Flow Duration': 'float64',
            'Flow Bytes/s': 'float64',
            'Flow Packets/s': 'float64',
        }

        print("Leyendo dataset...")
        df = dd.read_csv('/mnt/datos/AllCicMerged.csv', blocksize="100MB", dtype=tipos)

        # Limpiar cabeceras repetidas e Inf — una sola vez antes del bucle
        print("Limpiando Inf y Flow Duration <= 0...")
        df = df[df['label2'] != 'label2']
        df_clean = df.map_partitions(limpiar_particion)

        # Calcular target sobre benign ya limpio
        counts_clean = df_clean['label2'].value_counts().compute()
        print("\nClases limpias disponibles:")
        for label, n in counts_clean.items():
            print(f"  {label:<12} {n:>10,}")

        if 'benign' not in counts_clean.index:
            raise ValueError(f"'benign' no encontrado. Clases: {counts_clean.index.tolist()}")

        target_count = int(counts_clean['benign'])
        print(f"\nTarget (benign limpio): {target_count:,}")

        sampled_fragments = []

        for label, clean_count in counts_clean.items():
            class_df = df_clean[df_clean['label2'] == label]
            clean_count = int(clean_count)

            if clean_count == 0:
                print(f" - AVISO: {label} vacía tras limpieza, saltando.")
                continue

            if clean_count > target_count:
                # Clase grande: samplear exactamente target_count del pool limpio
                frac = target_count / clean_count
                print(f" - {label:<12} reduciendo {clean_count:>10,} → {target_count:,} ({frac*100:.1f}%)")
                sampled_fragments.append(class_df.sample(frac=frac, random_state=42))
            else:
                # Clase pequeña: todas las filas limpias disponibles
                print(f" - {label:<12} manteniendo {clean_count:>10,} filas limpias")
                sampled_fragments.append(class_df)

        print("\nConcatenando...")
        balanced_df = dd.concat(sampled_fragments)

        output_dir = '/mnt/datos/output/balanced_real_data/'
        os.makedirs(output_dir, exist_ok=True)

        print(f"Guardando en {output_dir}...")
        balanced_df.to_csv(output_dir + 'part-*.csv', index=False)
        print("\n¡Hecho! Resultado esperado:")
        print("  Clases grandes → exactamente target_count filas limpias")
        print("  Clases pequeñas → todas sus filas limpias (sin Inf)")

if __name__ == "__main__":
    balance_data()