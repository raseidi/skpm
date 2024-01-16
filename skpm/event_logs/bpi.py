import os

from .base import TUEventLog


class BPI12(TUEventLog):
    """BPI Challenge 2012 event log.

    DOI: 10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f

    This is an event log of a loan application process from a Dutch financial
    institute. The process is concerned with handling loan applications for
    private individuals. The data is enriched with decision and execution
    information added by a process mining tool.

    Args:
        root_folder (str, optional): Path where the event log will be stored.
            Defaults to "data/".
        save_as_pandas (bool, optional): Whether to save the event log as a pandas parquet file.
            Defaults to True.
        train_set (bool, optional): Whether to use the train set or the test set.
            If True, use the train set. If False, use the test set. Defaults to True.
    """

    url: str = "https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893"
    md5: str = "74c7ba9aba85bfcb181a22c9d565e5b5"
    file_name: str = "BPI_Challenge_2012.xes.gz"

    def preprocess(self):
        raise NotImplementedError


class BPI13ClosedProblems(TUEventLog):
    """BPI Challenge 2013 Closed problems."""

    url: str = "https://data.4tu.nl/file/1987a2a6-9f5b-4b14-8d26-ab7056b17929/8b99119d-9525-452e-bc8f-236ac76fa9c9"
    md5: str = "4f9c35942f42cb90d911ee4936bbad87"
    file_name: str = "BPI_Challenge_2013_closed_problems.xes.gz"

    def preprocess(self):
        raise NotImplementedError


class BPI13Incidents(TUEventLog):
    """BPI Challenge 2013 Incidents."""

    url: str = "https://data.4tu.nl/file/0fc5c579-e544-4fab-9143-fab1f5192432/aa51ffbb-25fd-4b5a-b0b8-9aba659b7e8c"
    md5: str = "d4809bd55e3e1c15b017ab4e58228297"
    file_name: str = "BPI_Challenge_2013_incidents.xes.gz"

    def preprocess(self):
        raise NotImplementedError


class BPI13Incidents(TUEventLog):
    """BPI Challenge 2013 open problems."""

    url: str = "https://data.4tu.nl/file/7aafbf5b-97ae-48ba-bd0a-4d973a68cd35/0647ad1a-fa73-4376-bdb4-1b253576c3a1"
    md5: str = "9663e544a2292edf1fe369747736e7b4"
    file_name: str = "BPI_Challenge_2013_open_problems.xes.gz"

    def preprocess(self):
        raise NotImplementedError


class BPI17(TUEventLog):
    """BPI Challenge 2013 open problems."""

    url: str = "https://data.4tu.nl/file/34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c"
    md5: str = "10b37a2f78e870d78406198403ff13d2"
    file_name: str = "BPI Challenge 2017.xes.gz"

    def preprocess(self):
        raise NotImplementedError


class BPI19(TUEventLog):
    """BPI Challenge 2019."""

    url: str = "https://data.4tu.nl/file/35ed7122-966a-484e-a0e1-749b64e3366d/864493d1-3a58-47f6-ad6f-27f95f995828"
    md5: str = "4eb909242351193a61e1c15b9c3cc814"
    file_name: str = "BPI_Challenge_2019.xes"

    def preprocess(self):
        raise NotImplementedError
