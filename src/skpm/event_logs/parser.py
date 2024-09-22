from lxml import etree
from itertools import chain
from joblib import Parallel, delayed
from typing import Generator


class Event(dict):
    pass


def parse_trace(trace: list[etree._Element]) -> list[Event]:
    """Parses a list of XML elements representing a trace.

    Args:
        trace (list[etree._Element]): List of XML elements representing a trace from a XES file.

    Returns:
        list[Event]: The respective events from the trace.
    """

    if type(trace) == bytes:
        trace = etree.fromstring(trace)

    case_attrs = Event()
    for attr in ["string", "date", "float", "boolean", "int"]:
        case_attrs.update(
            {f'case:{e.get("key")}': e.get("value") for e in trace.findall(attr)}
        )

    events = trace.findall("event")
    final_events = []
    for event in events:
        event_attrs = Event()
        for e in event.iter():
            if e.tag in ["string", "date", "float", "boolean", "int"]:
                event_attrs[e.get("key")] = e.get("value")
        event.clear()
        event_attrs.update(case_attrs)
        final_events.append(event_attrs)

    trace.clear()
    return final_events


def lazy_serialize(elements: list[etree._Element]) -> Generator[bytes]:
    """Lazy serialization of a list of XML elements. Used for parallel processing."""
    for element in elements:
        yield etree.tostring(element)


def read_from_xes(filepath: str, n_jobs: int = None) -> list[Event]:
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
    tree = etree.parse(filepath)
    traces = tree.findall("trace")

    if n_jobs in [1, None]:
        log = []
        for trace in traces:
            log.extend(parse_trace(trace))
    else:
        traces = lazy_serialize(traces)
        log = Parallel(n_jobs=n_jobs)(delayed(parse_trace)(trace) for trace in traces)
        log = list(chain(*log))
    return log
