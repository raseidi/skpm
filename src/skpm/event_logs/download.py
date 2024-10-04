import os
import typing as t
from urllib import request


def download_url(
    url: str, folder: t.Optional[str] = None, file_name: t.Optional[str] = None
) -> str:
    """Download a file from a `url` and place it in `folder`.

    Args:
        url (str): URL to download file from
        folder (str, optional): Folder to download file to.
            If None, use the current working directory. Defaults to None.
        file_name (str, optional): Name to save the file under.
            If None, use the basename of the URL. Defaults to None.

    Returns:
        folder (str): Path to downloaded file
    """
    if folder is None:
        folder = os.getcwd()

    if file_name is None:
        # TODO: maybe get the file_name from the request?
        # response.info().get_file_name()
        file_name = os.path.basename(url)
    path = os.path.join(folder, file_name)

    if os.path.exists(path):
        return path

    # try:
    os.makedirs(os.path.expanduser(os.path.normpath(folder)), exist_ok=True)
    # except OSError as e:
    #     raise e

    _urlretrieve(url=url, destination=path)
    return path


def _save_response_content(
    content: t.Iterator[bytes],
    destination: str,
) -> None:
    """
    Save the content received from an HTTP response to a file.

    Parameters
    ----------
    content : Iterator[bytes]
        Iterator yielding binary data chunks from the HTTP response.
    destination : str
        Path to the file where the content will be saved.

    Returns
    -------
    None
    """
    with open(destination, "wb") as fh:
        for chunk in content:
            # filter out keep-alive new chunks
            # if not chunk:
            #     continue

            fh.write(chunk)


def _urlretrieve(
    url: str, destination: str, chunk_size: int = 1024 * 32
) -> None:
    """
    Retrieve a URL and save its contents to a file.

    Parameters
    ----------
    url : str
        The URL of the resource to retrieve.
    destination : str
        Path to the file where the content will be saved.
    chunk_size : int, optional
        Size of the chunks to read from the response at a time, in bytes.
        Defaults to 32KB.

    Returns
    -------
    None
    """
    with request.urlopen(request.Request(url)) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""),
            destination,
        )
