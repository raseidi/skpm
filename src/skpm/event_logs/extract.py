import gzip
import os.path as osp
import zipfile


def extract_gz(path: str, folder: str):
    r"""Extracts a gz archive to a specific folder.

    Args:
        path (str): The path to the tar archive.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    path = osp.abspath(path)
    file_path = osp.join(folder, ".".join(path.split(".")[:-1]))
    with gzip.open(path, "r") as r:
        with open(file_path, "wb") as w:
            w.write(r.read())

    return file_path


# def extract_zip(path: str, folder: str):
#     r"""Extracts a zip archive to a specific folder.

#     Args:
#         path (str): The path to the tar archive.
#         folder (str): The folder.
#         log (bool, optional): If :obj:`False`, will not print anything to the
#             console. (default: :obj:`True`)
#     """
#     with zipfile.ZipFile(path, "r") as f:
#         f.extractall(folder)


# commenting out the following functions because
# they are not used in the codebase. Maybe in the future.

# import bz2
# import tarfile

# def extract_tar(path: str, folder: str, mode: str = 'r:gz'):
#     r"""Extracts a tar archive to a specific folder.

#     Args:
#         path (str): The path to the tar archive.
#         folder (str): The folder.
#         mode (str, optional): The compression mode. (default: :obj:`"r:gz"`)
#         log (bool, optional): If :obj:`False`, will not print anything to the
#             console. (default: :obj:`True`)
#     """
#     with tarfile.open(path, mode) as f:
#         f.extractall(folder)


# def extract_bz2(path: str, folder: str):
#     r"""Extracts a bz2 archive to a specific folder.

#     Args:
#         path (str): The path to the tar archive.
#         folder (str): The folder.
#         log (bool, optional): If :obj:`False`, will not print anything to the
#             console. (default: :obj:`True`)
#     """
#     path = osp.abspath(path)
#     with bz2.open(path, 'r') as r:
#         with open(osp.join(folder, '.'.join(path.split('.')[:-1])), 'wb') as w:
#             w.write(r.read())
