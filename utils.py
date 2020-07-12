import os
from shutil import rmtree


def check_folder(folder, remove_previous=True):
    if os.path.exists(folder):
        if remove_previous:
            rmtree(folder)
            os.makedirs(folder)
    else:
        os.makedirs(folder)


def abs_path(path, *paths):
    return os.path.join(path, *paths)
