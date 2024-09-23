from lxml import etree
from itertools import chain
from joblib import Parallel, delayed
from typing import Generator
import pandas as pd

DataFrame = pd.DataFrame


class Event(dict):
    pass

class TagXES:
    # attributes
    STRING: str = "string"
    DATE: str = "date"
    FLOAT: str = "float"
    BOOLEAN: str = "boolean"
    INT: str = "int"
    
    # elements
    EVENT: str = "event"
    TRACE: str = "trace"
    
    _DTYPES: tuple = (STRING, DATE, FLOAT, BOOLEAN, INT)
    
    @classmethod
    def is_attribute(cls, element: etree._Element) -> bool:
        """element is an attribute if it ends with one of the dtypes."""
        return element.tag.endswith(cls._DTYPES)
    
    @classmethod
    def is_valid(cls, element: etree._Element) -> bool:
        return element.tag.endswith(tuple(v for v in vars(cls).values() if not v.startswith("_")))
    
    @classmethod
    def get_dtypes(cls) -> tuple:
        return cls._DTYPES

tag: TagXES = TagXES

def extract_case_attributes(trace: etree._Element, ns: dict) -> Event:
    """
    Extracts case-level attributes from the trace.
    
    Using findall for case attributes is faster than using iter since
    cases has fewer attributes than events.

    Args:
        trace (etree._Element): The trace element.
        ns (dict): Namespace mapping for XML parsing.

    Returns:
        Event: A dictionary of case-level attributes.
    """
    case_attrs = Event()
    for attr in tag.get_dtypes():
        # Find all attributes of the given type in the trace
        attrs = trace.findall(attr, ns)
        # Update case_attrs with the found attributes
        case_attrs.update({
            f'case:{e.get("key")}': e.get("value") for e in attrs
        })
    return case_attrs


def extract_event_attributes(event: etree._Element) -> Event:
    """
    Extracts attributes from an event element.
    
    Using iter is slightly faster than findall for events since
    there many events and event attributes in a trace.

    Args:
        event (etree._Element): The event element.

    Returns:
        Event: A dictionary of event attributes.
    """
    event_attrs = Event()
    for e_attr in event.iter():
        if tag.is_attribute(e_attr):
            event_attrs[e_attr.get("key")] = e_attr.get("value")
    return event_attrs


def parse_trace(trace: list[etree._Element], ns: dict) -> list[Event]:
    """Parses a list of XML elements representing a trace.

    Args:
        trace (list[etree._Element]): List of XML elements representing a trace from a XES file.

    Returns:
        list[Event]: The respective events from the trace.
    """

    if type(trace) == bytes:
        trace = etree.fromstring(trace)

    case_attrs = extract_case_attributes(trace, ns)
    
    # Parse each event
    parsed_events = []
    events = trace.findall(tag.EVENT, ns)
    for event in events:
        event_attrs = extract_event_attributes(event)
        
        # Add case-level attributes to event attributes
        event_attrs.update(case_attrs)
        parsed_events.append(event_attrs)
        
        # Clear the event to free memory
        event.clear()

    trace.clear()
    return parsed_events


def lazy_serialize(elements: list[etree._Element]) -> Generator[bytes, None, None]:
    """Lazy serialization of a list of XML elements. Used for parallel processing."""
    for element in elements:
        yield etree.tostring(element)


def read_xes(filepath: str, n_jobs: int = None, as_df=True) -> DataFrame | list[Event]:
    """Reads an event log from a XES file.

    Rough overview:
        This function reads an event log from a XES file. It uses the lxml library to
        parse the XML file. The function is parallelized using the joblib library.

        1. For each trace, the function `_parse_trace` is called.
        2. Extract the case attributes and all the events from the trace.
        3. For each event, extract the event attributes.
        4. Update the event attributes with the case attributes.
        5. Append the event to the final list of events corresponding to the trace.
        6. Return trace and repeat.

    Args:
        filepath (str): Filepath to the XES file.
        n_jobs (int, optional): Number of CPU cores to use. If None, only one core
        is used. Defaults to None.

    Returns:
        list[Event]: an event log as a list of Event objects.
    """
    tree = etree.parse(filepath).getroot()
    ns = tree.nsmap
    
    
    traces = tree.findall(tag.TRACE, ns)

    if n_jobs in [1, None]:
        log = []
        for trace in traces:
            log.extend(parse_trace(trace, ns))
    else:
        traces = lazy_serialize(traces)
        log = Parallel(n_jobs=n_jobs)(delayed(parse_trace)(trace) for trace in traces)
        log = list(chain(*log))
        
    if as_df:
        log = pd.DataFrame(log)
        
    return log
