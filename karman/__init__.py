__version__ = '2.0'
import os
import torch
torch.set_default_dtype(torch.float32)

from .dataset import KarmanDataset
from .nn import SimpleNetwork
from .util import exponential_atmosphere, scale_density, date_to_index, normalize_time_series_data, get_ts_data, find_sw_from_thermo 
