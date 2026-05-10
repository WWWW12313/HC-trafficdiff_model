import torch

try:
    from icecream import install
    install()
except ImportError:
    pass

torch.set_num_threads(1)

from . import env  # noqa
from .data import *  # noqa
from .env import *  # noqa
from .metrics import *  # noqa
from .util import *  # noqa