import os
import sys
sys.path.append('../')
from src.directories import *
def numerize_directory(directory: str,type: str):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    files.sort()
    files_len = len(files)
    for idx, file in enumerate(files):

        file_extension = os.path.splitext(file)[1]
        new_file_name = f"{str(idx).zfill(files_len)}_{type}{file_extension}"

        print(file,new_file_name)
        #os.rename()
    

if __name__ == '__main__':
    print(TEXTURES_DIR)
    numerize_directory(TEXTURES_DIR,'texture')

