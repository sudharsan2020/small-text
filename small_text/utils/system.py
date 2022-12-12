import importlib
import os

TMP_DIR_VARIABLE = 'SMALL_TEXT_TEMP'


def get_tmp_dir_base():

    return os.environ[TMP_DIR_VARIABLE] if TMP_DIR_VARIABLE in os.environ else None


def is_pytorch_available():
    try:
        importlib.import_module('torch')
        return True
    except ImportError:
        return False


def is_transformers_available():
    try:
        importlib.import_module('transformers')
        return True
    except ImportError:
        return False
