import sys
from typing import List
from dotmap import DotMap

from .parser import Parser


def parse_args(sys_argv: List[str] = None):
    return Parser(sys_argv).parse()

conf = parse_args(sys.argv[1:])
args = DotMap(conf, _dynamic=False)
