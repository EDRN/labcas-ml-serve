# encoding: utf-8

'''🧠 JPL's LabCAS ML Service for EDRN: utilities.'''


import logging, argparse

_logger = logging.getLogger(__name__)


def add_logging_arguments(parser: argparse.ArgumentParser):
    '''Add a typical set of command-line logging options to the given ``parser``.'''
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--debug', action='store_const', dest='loglevel', const=logging.DEBUG, default=logging.INFO,
        help='🐞 Log copious and verbose messages suitable for developers'
    )
    group.add_argument(
        '--quiet', action='store_const', dest='loglevel', const=logging.WARNING,
        help="🤫 Don't log informational messages"
    )
