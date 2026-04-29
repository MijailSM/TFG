import sys
import os
    

def create(name):
    os.mkdir(f"./joblibs/{name}")
    os.mkdir(f"./joblibs/{name}/2clases")
    os.mkdir(f"./joblibs/{name}/8clases")
    os.mkdir(f"./joblibs/{name}/60clases")
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error in argumnents")
    name = sys.argv[1]
    create(name)