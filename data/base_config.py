
import sys


class BaseConfig:
    """
    Base class for configuration files of all kinds
    """

    def __init__(self):
        self.GPUS = [0,]

        self.SUBSET = "train"
        self.TEST_SUBSET = "test"
        self.VAL_SUBSET = "val"

        if sys.gettrace() is None:
            self.WORKERS = 16
        else:
            self.WORKERS = 0

        self.SIGMA = 2

        self.OPTIMIZER = "adam"

        self.VERBOSE = False

