from dataclasses import dataclass


@dataclass
class EventLogConfig:
    case_id: str = "case:concept:name"
    activity: str = "concept:name"
    resource: str = "org:resource"
    timestamp: str = "time:timestamp"

    default_file_format: str = ".parquet"

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)