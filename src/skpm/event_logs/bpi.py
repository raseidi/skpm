from skpm.event_logs.base import TUEventLog


class BPI12(TUEventLog):
    """BPI Challenge 2012 event log.

    This dataset is from the Business Process Intelligence (BPI) Challenge
    2012 and contains event logs from a real-life financial institution. The
    event log records the execution of various activities related to a loan
    application process. Each event in the log represents a step in handling a
    loan request, with relevant information about the case, timestamp, and
    resource involved.

    DOI: :doi:`BPI12 <10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f>`.

    Parameters
    ----------
        root_folder (str, optional): Path where the event log will be stored.
            Defaults to "data/".
        save_as_pandas (bool, optional): Whether to save the event log as a
        pandas parquet file.
            Defaults to True.
        train_set (bool, optional): Whether to use the train set or the test
        set.
            If True, use the train set. If False, use the test set. Defaults
            to True.

    Examples
    --------
    >>> bpi_12 = BPI12()
    >>> bpi_12.download()  # Manually download the event log
    >>> event_log = bpi_12.dataframe()  # Access the event log DataFrame
    """

    url: str = (
        "https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893"
    )
    md5: str = "74c7ba9aba85bfcb181a22c9d565e5b5"
    file_name: str = "BPI_Challenge_2012.xes.gz"

    _unbiased_split_params: dict = {
        "start_date": None,
        "end_date": "2012-02",
        "max_days": 32.28,
    }


class BPI13ClosedProblems(TUEventLog):
    """BPI Challenge 2013 Closed problems.

    The BPI 2013 Closed Problems log consists of 1487 cases and 6660 events.
    It originates from the problem management process of Volvo IT Belgium,
    focusing on cases where problems were diagnosed and resolved to enhance IT
    service quality.

    DOI: https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11


    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to
        True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train
        set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_13_incidents = BPI13Incidents()
    >>> bpi_13_incidents.download()  # Manually download the event log
    >>> event_log = bpi_13_incidents.dataframe()  # Access the event log DataFrame
    """

    url: str = (
        "https://data.4tu.nl/file/1987a2a6-9f5b-4b14-8d26-ab7056b17929/8b99119d-9525-452e-bc8f-236ac76fa9c9"
    )
    md5: str = "4f9c35942f42cb90d911ee4936bbad87"
    file_name: str = "BPI_Challenge_2013_closed_problems.xes.gz"


class BPI13Incidents(TUEventLog):
    """BPI Challenge 2013 Incidents.

    The BPI 2013 Incidents log contains 7554 cases and 65533 events.
    It is part of the incident management process at Volvo IT Belgium,
    aimed at restoring normal service operations for customers as quickly as
    possible, while maintaining high levels of service quality and
    availability.

    DOI: https://doi.org/10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee

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
    >>> event_log = bpi_13_open_problems.dataframe()  # Access the event log DataFrame
    """

    url: str = (
        "https://data.4tu.nl/file/0fc5c579-e544-4fab-9143-fab1f5192432/aa51ffbb-25fd-4b5a-b0b8-9aba659b7e8c"
    )
    md5: str = "d4809bd55e3e1c15b017ab4e58228297"
    file_name: str = "BPI_Challenge_2013_incidents.xes.gz"


class BPI13OpenProblems(TUEventLog):
    """BPI Challenge 2013 open problems.

    The BPI 2013 Open Problems log contains 819 cases and 2351 events.
    It originates from the problem management process of Volvo IT Belgium,
    focusing on unresolved problems that are still open and require further
    diagnosis and action to improve IT service quality.

    DOI: https://doi.org/10.4121/uuid:3537c19d-6c64-4b1d-815d-915ab0e479da

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to
        True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train
        set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_13_open_problems = BPI13OpenProblems()
    >>> bpi_13_open_problems.download()  # Manually download the event log
    >>> event_log = bpi_13_open_problems.dataframe()  # Access the event log DataFrame
    """

    url: str = (
        "https://data.4tu.nl/file/7aafbf5b-97ae-48ba-bd0a-4d973a68cd35/0647ad1a-fa73-4376-bdb4-1b253576c3a1"
    )
    md5: str = "9663e544a2292edf1fe369747736e7b4"
    file_name: str = "BPI_Challenge_2013_open_problems.xes.gz"


class BPI17(TUEventLog):
    """BPI Challenge 2017.

    The BPI 2017 event log originates from a loan application process at a
    Dutch financial institution. The data encompasses all loan applications
    submitted through an online system. This event log follows the same
    company and process as the BPI Challenge 2012. A notable feature of
    the new system is its ability to handle multiple offers for a single loan
    application, and these offers are tracked by their IDs within the event
    log.

    DOI: https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "./data".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to
        True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train
        set. If False, use the test set. Defaults to True.
    file_path : str, optional
        Path to the file containing the event log. If provided, the event log
        will be loaded from this file. Defaults to None.

    Examples
    --------
    >>> bpi_17 = BPI17()
    >>> bpi_17.download()  # Manually download the event log
    >>> event_log = bpi_17.dataframe()  # Access the event log DataFrame
    """

    url: str = (
        "https://data.4tu.nl/file/34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c"
    )
    md5: str = "10b37a2f78e870d78406198403ff13d2"
    file_name: str = "BPI Challenge 2017.xes.gz"

    _unbiased_split_params: dict = {
        "start_date": None,
        "end_date": "2017-01",
        "max_days": 47.81,
    }


class BPI19(TUEventLog):
    """BPI Challenge 2019.

    The BPI 2019 event log comes from a large multinational company in the
    coatings and paints industry, based in the Netherlands. It focuses on the
    purchase order handling process across 60 subsidiaries. Each purchase
    order contains one or more line items, with four types of matching flows:
    3-way matching with goods receipt, 3-way matching without goods receipt,
    2-way matching, and consignment. The log records 76,349 purchase documents,
    covering 251,734 items, with a total of 1,595,923 events. These events
    span 42 activities performed by 627 users, including both batch and normal
    users. The data is fully anonymized and structured in an IEEE-XES
    compliant format.

    DOI: https://data.4tu.nl/datasets/35ed7122-966a-484e-a0e1-749b64e3366d/1

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
    >>> event_log = bpi_19.dataframe()  # Access the event log DataFrame
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
    """BPI2020 Prepaid Travel Costs.

    The BPI 2020 Prepaid Travel Costs event log records two years of travel
    expense claims for a university. In 2017, the data covers two departments,
    while in 2018, it extends to the entire university. The dataset includes
    various declarations and requests, such as domestic and international
    travel declarations, pre-paid travel costs, and payment requests. The
    process begins with submission by an employee, followed by approval from
    the travel administration, budget owner, and supervisor. For international
    trips, prior permission from the supervisor is mandatory, while domestic
    trips do not require prior approval. Reimbursement claims can be filed
    either upon payment of costs or within two months after the trip.

    DOI: https://doi.org/10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to
        True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train
        set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_20 = BPI20PrepaidTravelCosts()
    >>> bpi_20.download()  # Manually download the event log
    >>> event_log = bpi_20.dataframe()  # Access the event log DataFrame
    """

    url: str = (
        "https://data.4tu.nl/file/fb84cf2d-166f-4de2-87be-62ee317077e5/612068f6-14d0-4a82-b118-1b51db52e73a"
    )
    md5: str = "b6ab8ee749e2954f09a4fef030960598"
    file_name: str = "PrepaidTravelCost.xes.gz"

    _unbiased_split_params: dict = {
        "start_date": None,
        "end_date": "2019-01",
        "max_days": 114.26,
    }


class BPI20TravelPermitData(TUEventLog):
    """BPI2020 Travel Permit Data.

    The BPI 2020 Travel Permit event log contains 7,065 cases and 86,581
    events, covering two years of travel expense claims at a university. In
    2017, data was gathered from two departments, expanding to the entire
    university in 2018. The log tracks the full process of travel permits,
    including related prepaid travel cost declarations and travel declarations.
    The process begins with the submission of a travel permit request by an
    employee, followed by approval from the travel administration, budget
    owner, and supervisor. For international trips, prior approval is required
    before making any travel arrangements, while domestic trips do not need
    prior approval. Reimbursement claims for costs can be submitted either
    upon payment or within two months after the trip.

    DOI: https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to
        True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train
        set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_20 = BPI20TravelPermitData()
    >>> bpi_20.download()  # Manually download the event log
    >>> event_log = bpi_20.dataframe()  # Access the event log DataFrame
    """

    url: str = (
        "https://data.4tu.nl/file/db35afac-2133-40f3-a565-2dc77a9329a3/12b48cc1-18a8-4089-ae01-7078fc5e8f90"
    )
    md5: str = "b6e9ff00d946f6ad4c91eb6fb550aee4"
    file_name: str = "PermitLog.xes.gz"

    _unbiased_split_params: dict = {
        "start_date": None,
        "end_date": "2019-10",
        "max_days": 258.81,
    }


class BPI20RequestForPayment(TUEventLog):
    """BPI2020 Request For Payment.

    The BPI 2020 Request for Payment event log contains 6,886 cases and 36,796
    events, primarily focusing on requests for payment that are not related to
    travel. However, some events may mistakenly be linked to travel, which is
    considered an unwanted deviation. The dataset covers two years of events,
    with data collected from two departments in 2017 and the entire university
    in 2018. The process for requests follows a similar flow to other
    declarations: submission by an employee, approval by the travel
    administration, and further approvals by the budget owner and supervisor
    if necessary.

    DOI: https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to
        True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train
        set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_20 = BPI20RequestForPayment()
    >>> bpi_20.download()  # Manually download the event log
    >>> event_log = bpi_20.dataframe()  # Access the event log DataFrame
    """

    url: str = (
        "https://data.4tu.nl/file/a6f651a7-5ce0-4bc6-8be1-a7747effa1cc/7b1f2e56-e4a8-43ee-9a09-6e64f45a1a98"
    )
    md5: str = "2eb4dd20e70b8de4e32cc3c239bde7f2"
    file_name: str = "RequestForPayment.xes.gz"

    _unbiased_split_params: dict = {
        "start_date": None,
        "end_date": "2018-12",
        "max_days": 28.86,
    }


class BPI20DomesticDeclarations(TUEventLog):
    """BPI2020 Domestic Declarations.

    The BPI 2020 Domestic Declarations event log contains 10,500 cases and
    56,437 events. The dataset focuses on domestic travel expense claims over
    a two-year period. In 2017, data was collected from two departments, while
    in 2018, it covered the entire university. Domestic declarations do not
    require prior permission; employees can complete these trips and later
    request reimbursement for the incurred costs. The process follows a
    similar approval flow: after submission by the employee, the request is
    reviewed by the travel administration and further approved by the budget
    owner and supervisor, if necessary.

    DOI: https://doi.org/10.4121/uuid:3f422315-ed9d-4882-891f-e180b5b4feb5

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to
        True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train
        set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_20 = BPI20DomesticDeclarations()
    >>> bpi_20.download()  # Manually download the event log
    >>> event_log = bpi_20.dataframe()  # Access the event log DataFrame
    """

    url: str = (
        "https://data.4tu.nl/file/6a0a26d2-82d0-4018-b1cd-89afb0e8627f/6eeb0328-f991-48c7-95f2-35033504036e"
    )
    md5: str = "6a78c39491498363ce4788e0e8ca75ef"
    file_name: str = "DomesticDeclarations.xes.gz"


class BPI20InternationalDeclarations(TUEventLog):
    """BPI2020 International Declarations.

    The BPI 2020 International Declarations event log contains 6,449 cases and
    72,151 events, covering two years of travel expense claims at a university.
    In 2017, the data was collected from two departments, expanding to the
    entire university in 2018. Unlike domestic trips, international trips
    require prior approval from the supervisor, which is obtained by
    submitting a travel permit. Once the permit is approved, the employee can
    proceed with travel arrangements. After the trip or payment of related
    expenses (e.g., flights or conference fees), a reimbursement claim is
    filed, which can be submitted either upon payment or within two months
    after the trip.

    DOI: https://doi.org/10.4121/uuid:2bbf8f6a-fc50-48eb-aa9e-c4ea5ef7e8c5

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to
        True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train
        set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> bpi_20 = BPI20InternationalDeclarations()
    >>> bpi_20.download()  # Manually download the event log
    >>> event_log = bpi_20.dataframe()  # Access the event log DataFrame

    """

    url: str = (
        "https://data.4tu.nl/file/91fd1fa8-4df4-4b1a-9a3f-0116c412378f/d45ee7dc-952c-4885-b950-4579a91ef426"
    )
    md5: str = "1ec65e046f70bb399cc6d2c154cd615a"
    file_name: str = "InternationalDeclarations.xes.gz"


class Sepsis(TUEventLog):
    """Sepsis.

    The Sepsis event log contains real-life hospital data regarding sepsis
    cases, a life-threatening condition often caused by infection. Each case
    in the log represents a patient's pathway through the hospital. The
    dataset includes around 1000 cases and approximately 15,000 events,
    covering 16 different activities. Additionally, 39 data attributes are
    recorded, such as the responsible group for each activity, test results,
    and information from checklists. All events and attribute values have been
    anonymized. While the timestamps of events have been randomized, the
    intervals between events within a trace remain unchanged.

    DOI:: https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460

    Parameters
    ----------
    root_folder : str, optional
        Path where the event log will be stored. Defaults to "data/".
    save_as_pandas : bool, optional
        Whether to save the event log as a pandas parquet file. Defaults to
        True.
    train_set : bool, optional
        Whether to use the train set or the test set. If True, use the train
        set. If False, use the test set. Defaults to True.

    Examples
    --------
    >>> sepsis = Sepsis()
    >>> sepsis.download()  # Manually download the event log
    >>> event_log = sepsis.dataframe()  # Access the event log DataFrame

    """

    url: str = (
        "https://data.4tu.nl/file/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/643dccf2-985a-459e-835c-a82bce1c0339"
    )

    md5: str = "b5671166ac71eb20680d3c74616c43d2"
    file_name: str = "Sepsis Cases - Event Log.xes.gz"
