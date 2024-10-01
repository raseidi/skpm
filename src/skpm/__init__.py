# read version from installed package
from importlib.metadata import version

__version__ = version("skpm")


from sklearn import set_config

set_config(transform_output="pandas")