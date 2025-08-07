import importlib.metadata
from os.path import abspath, dirname

__version__ = importlib.metadata.version("aviary-models")
PKG_DIR = dirname(abspath(__file__))
ROOT = dirname(PKG_DIR)
