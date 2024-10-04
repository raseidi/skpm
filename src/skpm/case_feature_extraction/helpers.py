import inspect
from sklearn.pipeline import Pipeline


def ensure_not_pipeline(fit_method):
    def wrapper(estimator, *args, **kwargs):
        in_pipeline = False
        for frame_info in inspect.stack():
            frame = frame_info.frame
            if "self" in frame.f_locals:
                caller_self = frame.f_locals["self"]
                if isinstance(caller_self, Pipeline):
                    in_pipeline = True
                    break
        if in_pipeline:
            class_name = estimator.__class__.__name__
            raise ValueError(
                f"{class_name} is a case-wise feature extractor and cannot be used in a pipeline."
            )

        return fit_method(estimator, *args, **kwargs)

    return wrapper
