import glob
import os
import sys


def numerize_directory(directory: str):

    directory_list = glob.glob(directory)
    directory_list.sort()

    num_files = len(directory_list)
    num_digits = len(str(num_files))

    for idx, filename in enumerate(directory_list):
        new_name = f"{str(idx + 1).zfill(num_digits)}{os.path.splitext(filename)[1]}"
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)

        os.rename(old_file, new_file)

        print(f"Renomeado com sucesso: {old_file} -> {new_file}")


if __name__ == "__main__":
    numerize_directory('../../data/textures')
