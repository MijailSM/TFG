import sys
import pandas as pd
import tarfile
import io

def main():
    if len(sys.argv) < 2:
        print("Usage: mergecsv [name1] [name2] ... [namen]")
        return
    
    csvs = []
    for name in sys.argv[1:]:
        if name.endswith('.tar.xz'):
            print(f'Descomprimiendo {name}...')
            with tarfile.open(name, "r:xz") as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith('.csv'))
                f = tar.extractfile(member)
                df = pd.read_csv(io.BytesIO(f.read()))
        else:
            df = pd.read_csv(name)
        
        print(f"Cargando archivo: {name} || Columnas: {len(df.columns)}")
            
        csvs.append(df)
    
    print("Uniendo todo...")
    merged = pd.concat(csvs, ignore_index=True)
    
    merged = merged.sample(frac=1).reset_index(drop=True)
    mname = "Merged_csv.csv"
    merged.to_csv(mname, index=False)
    print(f"Archivos unidos en {mname}")

if __name__ == "__main__":
    main()
    
    