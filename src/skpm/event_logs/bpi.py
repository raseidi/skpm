from skpm.event_logs.base import TUEventLog

class BPI11(TUEventLog):
    url: tuple[str] = (
        "https://data.4tu.nl/file/5ea5bb88-feaa-4e6f-a743-6460a755e05b/6f9640f9-0f1e-44d2-9495-ef9d1bd82218",
    )
    file_name: tuple[str] = ("Hospital_log.xes.gz",)

class BPI12(TUEventLog):
    """:doi:`BPI Challenge 2012 event log
    <10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f>`.

    This dataset is from the Business Process Intelligence (BPI) Challenge
    2012 and contains event logs from a real-life financial institution. The
    event log records the execution of various activities related to a loan
    application process. Each event in the log represents a step in handling a
    loan request, with relevant information about the case, timestamp, and
    resource involved.


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
    >>> event_log = bpi_12.dataframe  # Access the event log DataFrame
    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893",
    )
    file_name: tuple[str] = ("BPI_Challenge_2012.xes.gz",)
    unbiased_split_params: dict = {
        "start_date": None,
        "end_date": "2012-02",
        "max_days": 32.28,
    }


class BPI13ClosedProblems(TUEventLog):
    """:doi:`BPI Challenge 2013 Closed problems event log
    <https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11>`.

    The BPI 2013 Closed Problems log consists of 1487 cases and 6660 events.
    It originates from the problem management process of Volvo IT Belgium,
    focusing on cases where problems were diagnosed and resolved to enhance IT
    service quality.



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
    >>> event_log = bpi_13_incidents.dataframe  # Access the event log DataFrame
    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/1987a2a6-9f5b-4b14-8d26-ab7056b17929/8b99119d-9525-452e-bc8f-236ac76fa9c9",
    )
    file_name: tuple[str] = ("BPI_Challenge_2013_closed_problems.xes.gz",)


class BPI13Incidents(TUEventLog):
    """:doi:`BPI Challenge 2013 Incidents
    <https://doi.org/10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee>`.

    The BPI 2013 Incidents log contains 7554 cases and 65533 events.
    It is part of the incident management process at Volvo IT Belgium,
    aimed at restoring normal service operations for customers as quickly as
    possible, while maintaining high levels of service quality and
    availability.

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
    >>> event_log = bpi_13_open_problems.dataframe  # Access the event log DataFrame
    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/0fc5c579-e544-4fab-9143-fab1f5192432/aa51ffbb-25fd-4b5a-b0b8-9aba659b7e8c",
    )
    file_name: tuple[str] = ("BPI_Challenge_2013_incidents.xes.gz",)


class BPI13OpenProblems(TUEventLog):
    """:doi:`BPI Challenge 2013 open problems
    <https://doi.org/10.4121/uuid:3537c19d-6c64-4b1d-815d-915ab0e479da>`.

    The BPI 2013 Open Problems log contains 819 cases and 2351 events.
    It originates from the problem management process of Volvo IT Belgium,
    focusing on unresolved problems that are still open and require further
    diagnosis and action to improve IT service quality.


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
    >>> event_log = bpi_13_open_problems.dataframe  # Access the event log DataFrame
    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/7aafbf5b-97ae-48ba-bd0a-4d973a68cd35/0647ad1a-fa73-4376-bdb4-1b253576c3a1",
    )
    file_name: tuple[str] = ("BPI_Challenge_2013_open_problems.xes.gz",)


class BPI15(TUEventLog):
    """:doi:`BPI Challenge 2015 M5
    <https://doi.org/10.4121/uuid:b32c6fe5-f212-4286-9774-58dd53511cf8>`.

    The BPI 2015 event log is composed of five sub-logs (municipality 1-5).
    This class concatenates these sub-logs into a single event log.

    This data is provided by five Dutch municipalities. 
    The data contains all building permit applications over a period of 
    approximately four years. There are many different activities present, 
    denoted by both codes (attribute concept:name) and labels, both in Dutch 
    (attribute taskNameNL) and in English (attribute taskNameEN). The cases 
    in the log contain information on the main application as well as objection 
    procedures in various stages. Furthermore, information is available about 
    the resource that carried out the task and on the cost of the application 
    (attribute SUMleges). The processes in the five municipalities should be 
    identical, but may differ slightly. Especially when changes are made to 
    procedures, rules or regulations the time at which these changes are 
    pushed into the five municipalities may differ. Of course, over the four 
    year period, the underlying processes have changed. The municipalities 
    have a number of questions, namely: What are the roles of the people 
    involved in the various stages of the process and how do these roles 
    differ across municipalities? What are the possible points for improvement 
    on the organizational structure for each of the municipalities? The 
    employees of two of the five municipalities have physically moved into 
    the same location recently. Did this lead to a change in the processes 
    and if so, what is different? Some of the procedures will be outsourced 
    from 2018, i.e. they will be removed from the process and the applicant 
    needs to have these activities performed by an external party before 
    submitting the application. What will be the effect of this on the 
    organizational structures in the five municipalities? Where are differences 
    in throughput times between the municipalities and how can these be 
    explained? What are the differences in control flow between the 
    municipalities? There are five different log files available in this 
    collection. Events are labeled with both a code and a Dutch and English 
    label. Each activity code consists of three parts: two digits, a variable 
    number of characters, and then three digits. The first two digits as well 
    as the characters indicate the subprocess the activity belongs to. For 
    instance '01_HOOFD_xxx' indicates the main process and '01_BB_xxx' 
    indicates the 'objections and complaints' ('Beroep en Bezwaar' in Dutch) 
    subprocess. The last three digits hint on the order in which activities 
    are executed, where the first digit often indicates a phase within a 
    process. Each trace and each event, contain several data attributes that 
    can be used for various checks and predictions. Furthermore, some employees 
    may have performed tasks for different municipalities, i.e. if the employee 
    number is the same, it is safe to assume the same person is being 
    identified.


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
    >>> bpi_15 = BPI15()
    >>> bpi_15.download()  # Manually download the event log
    >>> event_log = bpi_15.dataframe  # Access the event log DataFrame
    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/32b70553-0765-4808-b155-aa5319802c8a/d39e1365-e4b8-4cb8-83d3-0b01cbf6f8c2",
        "https://data.4tu.nl/file/34216d8a-f054-46d4-bf03-d9352f90967e/68923819-b085-43be-abe2-e084a0f1381f",
        "https://data.4tu.nl/file/d6741425-5f62-4a59-92c5-08bae64b4611/21b574ab-02ba-4dfb-badc-bb46ce0edc44",
        "https://data.4tu.nl/file/372d0cad-3fb1-4627-8ea9-51a09923d331/d653a8ec-4cd1-4029-8b61-6cfde4f4a666",
        "https://data.4tu.nl/file/6f35269e-4ce7-4bc4-9abb-b3cea04cad00/2c8d5827-3e08-471d-98e2-6ffdec92f958",
    )
    file_name: tuple[str] = (
        "BPIC15_5.xes",
        "BPIC15_4.xes",
        "BPIC15_3.xes",
        "BPIC15_2.xes",
        "BPIC15_1.xes",
    )
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        cols_to_drop = ["activityNameNL", "concept:name",]
        self._dataframe = self._dataframe.drop(columns=cols_to_drop, errors="ignore")
        self._dataframe = self._dataframe.rename(columns={"activityNameEN": "concept:name",})
        

class BPI17(TUEventLog):
    """:doi:`BPI Challenge 2017
    <https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b>`.

    The BPI 2017 event log originates from a loan application process at a
    Dutch financial institution. The data encompasses all loan applications
    submitted through an online system. This event log follows the same
    company and process as the BPI Challenge 2012. A notable feature of
    the new system is its ability to handle multiple offers for a single loan
    application, and these offers are tracked by their IDs within the event
    log.


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
    >>> event_log = bpi_17.dataframe  # Access the event log DataFrame
    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c",
    )
    file_name: tuple[str] = ("BPI Challenge 2017.xes.gz",)
    unbiased_split_params: dict = {
        "start_date": None,
        "end_date": "2017-01",
        "max_days": 47.81,
    }


class BPI19(TUEventLog):
    """:doi:`BPI Challenge 2019
    <https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1>`.


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
    >>> event_log = bpi_19.dataframe  # Access the event log DataFrame
    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/35ed7122-966a-484e-a0e1-749b64e3366d/864493d1-3a58-47f6-ad6f-27f95f995828",
    )
    file_name: tuple[str] = ("BPI_Challenge_2019.xes",)
    unbiased_split_params: dict = {
        "start_date": "2018-01",
        "end_date": "2019-02",
        "max_days": 143.33,
    }


class BPI20PrepaidTravelCosts(TUEventLog):
    """:doi:`BPI2020 Prepaid Travel Costs
    <https://doi.org/10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165>`.


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
    >>> event_log = bpi_20.dataframe  # Access the event log DataFrame
    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/fb84cf2d-166f-4de2-87be-62ee317077e5/612068f6-14d0-4a82-b118-1b51db52e73a",
    )
    file_name: tuple[str] = ("PrepaidTravelCost.xes.gz",)
    unbiased_split_params: dict = {
        "start_date": None,
        "end_date": "2019-01",
        "max_days": 114.26,
    }


class BPI20TravelPermitData(TUEventLog):
    """:doi:`BPI2020 Travel Permit Data
    <https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550>`.

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
    >>> event_log = bpi_20.dataframe  # Access the event log DataFrame
    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/db35afac-2133-40f3-a565-2dc77a9329a3/12b48cc1-18a8-4089-ae01-7078fc5e8f90",
    )
    file_name: tuple[str] = ("PermitLog.xes.gz",)
    unbiased_split_params: dict = {
        "start_date": None,
        "end_date": "2019-10",
        "max_days": 258.81,
    }


class BPI20RequestForPayment(TUEventLog):
    """:doi:`BPI2020 Request For Payment
    <https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030>`.


    The BPI 2020 Request for Payment event log contains 6,886 cases and 36,796
    events, primarily focusing on requests for payment that are not related to
    travel. However, some events may mistakenly be linked to travel, which is
    considered an unwanted deviation. The dataset covers two years of events,
    with data collected from two departments in 2017 and the entire university
    in 2018. The process for requests follows a similar flow to other
    declarations: submission by an employee, approval by the travel
    administration, and further approvals by the budget owner and supervisor
    if necessary.


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
    >>> event_log = bpi_20.dataframe  # Access the event log DataFrame
    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/a6f651a7-5ce0-4bc6-8be1-a7747effa1cc/7b1f2e56-e4a8-43ee-9a09-6e64f45a1a98",
    )
    file_name: tuple[str] = ("RequestForPayment.xes.gz",)
    unbiased_split_params: dict = {
        "start_date": None,
        "end_date": "2018-12",
        "max_days": 28.86,
    }


class BPI20DomesticDeclarations(TUEventLog):
    """:doi:`BPI2020 Domestic Declarations
    <https://doi.org/10.4121/uuid:3f422315-ed9d-4882-891f-e180b5b4feb5>`.


    The BPI 2020 Domestic Declarations event log contains 10,500 cases and
    56,437 events. The dataset focuses on domestic travel expense claims over
    a two-year period. In 2017, data was collected from two departments, while
    in 2018, it covered the entire university. Domestic declarations do not
    require prior permission; employees can complete these trips and later
    request reimbursement for the incurred costs. The process follows a
    similar approval flow: after submission by the employee, the request is
    reviewed by the travel administration and further approved by the budget
    owner and supervisor, if necessary.


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
    >>> event_log = bpi_20.dataframe  # Access the event log DataFrame
    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/6a0a26d2-82d0-4018-b1cd-89afb0e8627f/6eeb0328-f991-48c7-95f2-35033504036e",
    )
    file_name: tuple[str] = ("DomesticDeclarations.xes.gz",)


class BPI20InternationalDeclarations(TUEventLog):
    """:doi:`BPI2020 International Declarations
    <https://doi.org/10.4121/uuid:2bbf8f6a-fc50-48eb-aa9e-c4ea5ef7e8c5>`.

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
    >>> event_log = bpi_20.dataframe  # Access the event log DataFrame

    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/91fd1fa8-4df4-4b1a-9a3f-0116c412378f/d45ee7dc-952c-4885-b950-4579a91ef426",
    )
    file_name: tuple[str] = ("InternationalDeclarations.xes.gz",)


class Sepsis(TUEventLog):
    """:doi:`Sepsis
    <https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460>`.


    The Sepsis event log contains real-life hospital data regarding sepsis
    cases, a life-threatening condition often caused by infection. Each case
    in the log represents a patient's pathway through the hospital. The
    dataset includes around 1000 cases and approximately 15,000 events,
    covering 16 different activities. Additionally, 39 data attributes are
    recorded, such as the responsible group for each activity, test results,
    and information from checklists. All events and attribute values have been
    anonymized. While the timestamps of events have been randomized, the
    intervals between events within a trace remain unchanged.


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
    >>> event_log = sepsis.dataframe  # Access the event log DataFrame

    """

    url: tuple[str] = (
        "https://data.4tu.nl/file/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/643dccf2-985a-459e-835c-a82bce1c0339",
    )
    file_name: tuple[str] = ("Sepsis Cases - Event Log.xes.gz",)
