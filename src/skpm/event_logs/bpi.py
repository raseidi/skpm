from skpm.event_logs.base import TUEventLog

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

    url: str = (
        "https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893"
    )
    md5: str = "74c7ba9aba85bfcb181a22c9d565e5b5"
    file_name: str = "BPI_Challenge_2012.xes.gz"

    _unbiased_split_params: dict = {    
        "start_date": None ,
        "end_date": "2012-02",
        "max_days": 32.28,
    }

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

    url: str = (
        "https://data.4tu.nl/file/1987a2a6-9f5b-4b14-8d26-ab7056b17929/8b99119d-9525-452e-bc8f-236ac76fa9c9"
    )
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

    url: str = (
        "https://data.4tu.nl/file/0fc5c579-e544-4fab-9143-fab1f5192432/aa51ffbb-25fd-4b5a-b0b8-9aba659b7e8c"
    )
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

    url: str = (
        "https://data.4tu.nl/file/7aafbf5b-97ae-48ba-bd0a-4d973a68cd35/0647ad1a-fa73-4376-bdb4-1b253576c3a1"
    )
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

    url: str = (
        "https://data.4tu.nl/file/34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c"
    )
    md5: str = "10b37a2f78e870d78406198403ff13d2"
    file_name: str = "BPI Challenge 2017.xes.gz"
    
    _unbiased_split_params: dict = {
        "start_date": None ,
        "end_date": "2017-01",
        "max_days": 47.81,
    }

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

    url: str = (
        "https://data.4tu.nl/file/35ed7122-966a-484e-a0e1-749b64e3366d/864493d1-3a58-47f6-ad6f-27f95f995828"
    )
    md5: str = "4eb909242351193a61e1c15b9c3cc814"
    file_name: str = "BPI_Challenge_2019.xes"

    _unbiased_split_params: dict = {
        "start_date": "2018-01",
        "end_date": "2019-02",
        "max_days": 143.33,
    }

class BPI20PrepaidTravelCosts(TUEventLog):
    
    url: str = (
        "https://data.4tu.nl/file/fb84cf2d-166f-4de2-87be-62ee317077e5/612068f6-14d0-4a82-b118-1b51db52e73a"
    )
    md5: str = "b6ab8ee749e2954f09a4fef030960598"
    file_name: str = "PrepaidTravelCost.xes.gz"

    _unbiased_split_params: dict = {
        "start_date": None ,
        "end_date": "2019-01",
        "max_days": 114.26,
    }

class BPI20TravelPermitData(TUEventLog):
    url: str = (
        "https://data.4tu.nl/file/db35afac-2133-40f3-a565-2dc77a9329a3/12b48cc1-18a8-4089-ae01-7078fc5e8f90"
    )
    md5: str = "b6e9ff00d946f6ad4c91eb6fb550aee4"
    file_name: str = "PermitLog.xes.gz"

    _unbiased_split_params: dict = {
        "start_date": None ,
        "end_date": "2019-10",
        "max_days": 258.81,
    }

class BPI20RequestForPayment(TUEventLog):
    url: str = (
        "https://data.4tu.nl/file/a6f651a7-5ce0-4bc6-8be1-a7747effa1cc/7b1f2e56-e4a8-43ee-9a09-6e64f45a1a98"
    )
    md5 : str = "2eb4dd20e70b8de4e32cc3c239bde7f2"
    file_name: str = "RequestForPayment.xes.gz"
    
    _unbiased_split_params: dict = {
        "start_date": None ,
        "end_date": "2018-12",
        "max_days": 28.86,
    }

class BPI20DomesticDeclarations(TUEventLog):
    url: str = (
        "https://data.4tu.nl/file/6a0a26d2-82d0-4018-b1cd-89afb0e8627f/6eeb0328-f991-48c7-95f2-35033504036e"
    )
    md5: str = "6a78c39491498363ce4788e0e8ca75ef"
    file_name: str = "DomesticDeclarations.xes.gz"

class BPI20InternationalDeclarations(TUEventLog):
    url: str = (
        "https://data.4tu.nl/file/91fd1fa8-4df4-4b1a-9a3f-0116c412378f/d45ee7dc-952c-4885-b950-4579a91ef426"
    )
    md5: str = "1ec65e046f70bb399cc6d2c154cd615a"
    file_name: str = "InternationalDeclarations.xes.gz"

class Sepsis(TUEventLog):
    url: str = (
        "https://data.4tu.nl/file/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/643dccf2-985a-459e-835c-a82bce1c0339"
    )
    
    md5: str = "b5671166ac71eb20680d3c74616c43d2"
    file_name: str = "Sepsis Cases - Event Log.xes.gz"
