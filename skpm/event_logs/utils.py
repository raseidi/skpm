import os
from typing import Iterator, Optional
from urllib import error, request
import tarfile
import zipfile


def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
) -> None:
    with open(destination, "wb") as fh:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024 * 32) -> None:
    with request.urlopen(request.Request(url)) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""),
            filename,
            length=response.length,
        )


def download_url(
    url: str, root: str, filename: Optional[str] = None, max_redirect_hops: int = 3
) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    file_path = os.path.join(root, filename)

    try:
        print("Downloading " + url + " to " + file_path)
        _urlretrieve(url=url, filename=file_path)
    except (error.URLError, OSError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print(
                "Failed download. Trying https -> http instead. Downloading "
                + url
                + " to "
                + file_path
            )
            _urlretrieve(url, file_path)
        else:
            raise e


def extract_data(from_path: str, to_path: Optional[str] = None) -> None:
    """
    Extracts data from a compressed file (zip or tar) to a given path.
    If no path is given, it will extract to the same folder as the compressed file.

    Args:
        from_path (str): Path to the compressed file
        to_path (str, optional): Path to extract the file. Defaults to None.
    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    temp = os.path.join(to_path, "temp")
    os.mkdir(temp)
    if tarfile.is_tarfile(from_path):
        with tarfile.open(from_path, "r:*") as tar:
            tar.extractall(temp)
    elif zipfile.is_zipfile(from_path):
        with zipfile.ZipFile(from_path, "r") as zip:
            zip.extractall(temp)
    else:
        raise ValueError(f"Unknown file format: {from_path}")

    # cleaning temp folder and renaming extracted file
    os.remove(from_path)

    # cleaning temp folder created on `extract_data` function
    extracted = os.listdir(temp)
    filename = os.path.basename(from_path)
    if len(extracted) == 1:
        file, extension = extracted[0].split(".")
        os.rename(
            os.path.join(temp, extracted[0]),
            os.path.join(to_path, filename + "." + extension),
        )
    else:  # move from temp to root
        for file in os.listdir(temp):
            os.rename(os.path.join(temp, extracted[0]), os.path.join(to_path, file))
    os.rmdir(temp)


def download_and_extract_archive(
    url: str,
    root: str,
    filename: Optional[str] = None,
) -> None:
    """
    Modified from torchvision.datasets.utils.download_and_extract_archive
    It downloads and extracts an archive file from a URL.

    Args:
        url (str): URL to download the file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """
    root = os.path.expanduser(root)

    if not filename:
        filename = os.path.basename(url)

    os.makedirs(root, exist_ok=True)
    download_url(url, root, filename)

    downloaded_file = os.path.join(root, filename)
    print(f"Extracting {downloaded_file} to {root}")
    extract_data(from_path=downloaded_file, to_path=root)
