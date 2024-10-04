import os
from skpm.event_logs.extract import extract_gz
from skpm.event_logs.download import download_url


def _download(test_folder: str):
    url = "https://data.4tu.nl/file/1987a2a6-9f5b-4b14-8d26-ab7056b17929/8b99119d-9525-452e-bc8f-236ac76fa9c9"
    file_name = "BPI_Challenge_2013_closed_problems.xes.gz"
    output_fold_download = download_url(
        url, folder=test_folder, file_name=file_name
    )
    exists = os.path.exists(output_fold_download)
    assert exists

    output_fold_extract = extract_gz(
        path=output_fold_download, folder=os.path.dirname(output_fold_download)
    )
    extracted_exists = os.path.exists(output_fold_download.replace(".gz", ""))
    assert extracted_exists

    duplicated = download_url(url, folder=test_folder, file_name=file_name)
    assert duplicated == output_fold_download

    no_file_name = download_url(url, folder=".", file_name=None)
    assert os.path.isfile(no_file_name)
    os.remove(no_file_name)

    if exists:
        base_output_fold_download = os.path.abspath(
            os.path.dirname(output_fold_download)
        )
        if base_output_fold_download != os.getcwd():
            import shutil

            shutil.rmtree(base_output_fold_download)
        else:
            os.remove(output_fold_download)
            os.remove(output_fold_extract)


def test_download_extract():
    _download(test_folder="test_download_skpm")
    _download(test_folder=None)
    _download(test_folder=".")