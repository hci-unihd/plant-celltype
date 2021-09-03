from plantcelltype.graphnn.trainer import train
from plantcelltype.utils.io import load_yaml

from plantcelltype.utils.utils import parser

_args = parser()
_config = load_yaml(_args.config)
train(_config)
