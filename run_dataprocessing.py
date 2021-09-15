from plantcelltype.run_pipeline import main
from plantcelltype.utils.io import load_yaml

from plantcelltype.utils.utils import parser

if __name__ == '__main__':
    _args = parser()
    _config = load_yaml(_args.config)
    main(_config)
