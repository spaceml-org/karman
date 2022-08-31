__version__ = '0.0.4'

from .dataset import ThermosphericDensityDataset
from .nn import FeedForwardDensityPredictor, LSTMPredictor, FullFeatureDensityPredictor, Fism2DailyDensityPredictor, OmniDensityPredictor, Fism2FlareDensityPredictor, CNNDensityPredictor
from .benchmark import Benchmark
from .nn import FullFeatureFeedForward, NoFism2FlareFeedForward, NoFism2DailyFeedForward, NoOmniFeedForward, NoFism2FlareAndDailyFeedForward
