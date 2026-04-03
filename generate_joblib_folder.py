import sys
import os

if len(sys.argv) != 2:
    print("Error in argumnents")
    
name = sys.argv[1]

os.mkdir(f"./joblibs/{name}")
os.mkdir(f"./joblibs/{name}/2clases")
os.mkdir(f"./joblibs/{name}/8clases")
os.mkdir(f"./joblibs/{name}/60clases")