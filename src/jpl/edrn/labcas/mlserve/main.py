# encoding: utf-8

'''🧠 JPL's LabCAS ML Service for EDRN: main entrypoint and setup.'''


from . import VERSION
from .utils import add_logging_arguments
import sys, argparse, logging


__version__ = VERSION
_logger = logging.getLogger(__name__)


def main():
    '''Entrypoint: parse args and start things up.'''
    parser = argparse.ArgumentParser(description="JPL's LabCAS Machine Language Service for EDRN")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    add_logging_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel, format='%(levelname)s %(message)s')
    _logger.debug('Parsed args = %r', args)
    sys.exit(0)


if __name__ == '__main__':
    main()
