import os

def numerize_directory(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    files.sort()

    for idx, file in enumerate(files):

        file_extension = os.path.splitext(file)[1]
        new_file_name = f"{idx}"

