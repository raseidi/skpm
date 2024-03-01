from .base import TUEventLog


class BPI12(TUEventLog):
    """BPI Challenge 2012 event log.

    DOI: 10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f

    This is an event log of a loan application process from a Dutch financial
    institute. The process is concerned with handling loan applications for
    private individuals. The data is enriched with decision and execution
    information added by a process mining tool.

    Parameters
    ----------
        root_folder (str, optional): Path where the event log will be stored.
            Defaults to "data/".
        save_as_pandas (bool, optional): Whether to save the event log as a pandas parquet file.
            Defaults to True.
        train_set (bool, optional): Whether to use the train set or the test set.
            If True, use the train set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_12 = BPI12()
    >>> bpi_12.download()  # Manually download the event log
    >>> event_log = bpi_12.log  # Access the event log DataFrame
    """
    url: str = "https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893"
    md5: str = "74c7ba9aba85bfcb181a22c9d565e5b5"
    file_name: str = "BPI_Challenge_2012.xes.gz"


class BPI13ClosedProblems(TUEventLog):
    """
    BPI Challenge 2013 Closed problems.

    DOI: 10.4121/uuid:c7d29566-0d71-4de5-b18a-5b68f989b3f9

    The BPI Challenge 2013 focused on analyzing event data related to incidents, problems, and changes in an information technology (IT) service management context.
    The closed problems dataset provided for the challenge contains information about resolved issues and problems within an IT service management system.

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_13_incidents = BPI13Incidents()
    >>> bpi_13_incidents.download()  # Manually download the event log
    >>> event_log = bpi_13_incidents.log  # Access the event log DataFrame
    """
    url: str = "https://data.4tu.nl/file/1987a2a6-9f5b-4b14-8d26-ab7056b17929/8b99119d-9525-452e-bc8f-236ac76fa9c9"
    md5: str = "4f9c35942f42cb90d911ee4936bbad87"
    file_name: str = "BPI_Challenge_2013_closed_problems.xes.gz"


class BPI13Incidents(TUEventLog):
    """
    BPI Challenge 2013 Incidents.

    DOI: 10.4121/uuid:d4b0e8b3-10ec-41cb-bf21-c4a34815c3c8

    The BPI Challenge 2013 involved analyzing event data from an IT service management system, focusing on incidents, problems, and changes.
    The incidents dataset provided for the challenge contains information about various incidents reported within the IT service management context.

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_13_open_problems = BPI13OpenProblems()
    >>> bpi_13_open_problems.download()  # Manually download the event log
    >>> event_log = bpi_13_open_problems.log  # Access the event log DataFrame
    """

    url: str = "https://data.4tu.nl/file/0fc5c579-e544-4fab-9143-fab1f5192432/aa51ffbb-25fd-4b5a-b0b8-9aba659b7e8c"
    md5: str = "d4809bd55e3e1c15b017ab4e58228297"
    file_name: str = "BPI_Challenge_2013_incidents.xes.gz"


class BPI13OpenProblems(TUEventLog):
    """
    BPI Challenge 2013 open problems.

    DOI: 10.4121/uuid:89a55e11-ba94-47f7-b76a-1bc79299ed6c

    The BPI Challenge 2013 provided datasets related to incidents, problems, and changes in IT service management.
    The open problems dataset contains information about unresolved issues and problems within an IT service management system.

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_13_open_problems = BPI13OpenProblems()
    >>> bpi_13_open_problems.download()  # Manually download the event log
    >>> event_log = bpi_13_open_problems.log  # Access the event log DataFrame
    """
    url: str = "https://data.4tu.nl/file/7aafbf5b-97ae-48ba-bd0a-4d973a68cd35/0647ad1a-fa73-4376-bdb4-1b253576c3a1"
    md5: str = "9663e544a2292edf1fe369747736e7b4"
    file_name: str = "BPI_Challenge_2013_open_problems.xes.gz"


class BPI17(TUEventLog):
    """
    BPI Challenge 2017.



    DOI: 10.4121/uuid:d444d7f1-f81b-45c5-9a1a-0f4d1c1cee7e

    The BPI Challenge 2017 focused on analyzing event data related to business processes in various domains.

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "./data".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train set. If False, use the test set. Defaults to True.
    file_path : str, optional
        Path to the file containing the event log. If provided, the event log will be loaded from this file. Defaults to None.

    Notes
    -----
    The BPI Challenge 2017 event log contains data related to open problems in business processes.
    The event log is available for download from the provided URL.
    The MD5 hash is used for integrity verification of the downloaded file.

    Raises
    ------
    NotImplementedError
        This class does not support direct downloading of the event log from 4TU. Manual download is required.

    Examples
    --------
    >>> bpi_17 = BPI17()
    >>> bpi_17.download()  # Manually download the event log
    >>> event_log = bpi_17.log  # Access the event log DataFrame
    """
    url: str = "https://data.4tu.nl/file/34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c"
    md5: str = "10b37a2f78e870d78406198403ff13d2"
    file_name: str = "BPI Challenge 2017.xes.gz"


class BPI19(TUEventLog):
    """
    BPI Challenge 2019.

    DOI: 10.4121/uuid:22e7d7a6-7a1c-4ea4-9b45-7a7d7a89835b

    The BPI Challenge 2019 focused on analyzing event data related to business processes in various domains.
    Participants were provided with event logs representing real-world processes, and the challenge involved analyzing these logs to gain insights and propose process improvements.

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_19 = BPI19()
    >>> bpi_19.download()  # Manually download the event log
    >>> event_log = bpi_19.log  # Access the event log DataFrame
    """
    url: str = "https://data.4tu.nl/file/35ed7122-966a-484e-a0e1-749b64e3366d/864493d1-3a58-47f6-ad6f-27f95f995828"
    md5: str = "4eb909242351193a61e1c15b9c3cc814"
    file_name: str = "BPI_Challenge_2019.xes"


class BPI20(TUEventLog):
    """
    BPI Challenge 2020.

    DOI: 10.4121/uuid:62b0313f-3bde-447f-a4ce-f28aa06f49b2

    The BPI Challenge 2020 focused on analyzing event data related to business processes in various domains.
    This class provides access to different versions of the BPI Challenge 2020 event log.

    Parameters
    ----------
    version : str, optional
        The version of the event log to use. Available versions are "permit", "request", "domestic", "prepaid", and "international". Defaults to "permit".
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "./data".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train set. If False, use the test set. Defaults to True.
    file_path : str, optional
        Path to the file containing the event log. If provided, the event log will be loaded from this file. Defaults to None.

    Notes
    -----
    The BPI Challenge 2020 event log is available in multiple versions corresponding to different types of processes (permit, request, domestic, prepaid, international).
    Each version has its own URL, MD5 hash, and file name.

    Raises
    ------
    NotImplementedError
        This class does not support direct downloading of the event log from 4TU. Manual download is required.
    ValueError
        If an unsupported version is specified.

    Examples
    --------
    >>> bpi_20_permit = BPI20(version="permit")
    >>> bpi_20_permit.download()  # Manually download the event log
    >>> event_log = bpi_20_permit.log  # Access the event log DataFrame
    """

    versions = ("permit", "request", "domestic", "prepaid", "international")
    urls: dict = {
        "permit": "url1",
        "request": "url2",
        "domestic": "url3",
        "prepaid": "url4",
        "international": "url5",
    }
    md5s: dict = {
        "permit": "md51",
        "request": "md52",
        "domestic": "md53",
        "prepaid": "md54",
        "international": "md55",
    }
    file_names: dict = {
        "permit": "BPI_Challenge_2020_permit.xes.gz",
        "request": "BPI_Challenge_2020_request.xes.gz",
        "domestic": "BPI_Challenge_2020_domestic.xes.gz",
        "prepaid": "BPI_Challenge_2020_prepaid.xes.gz",
        "international": "BPI_Challenge_2020_international.xes.gz",
    }

    def __init__(
            self,
            version: str = "permit",
            root_folder: str = "./data",
            save_as_pandas: bool = True,
            train_set: bool = True,
            file_path: str = None,
    ) -> None:
        raise NotImplementedError("TODO: download from 4TU")

        if version not in self.versions:
            raise ValueError(f"Version {version} not in {self.versions}")
        # TODO

        super().__init__(root_folder, save_as_pandas, train_set, file_path)
