"""Generator registry."""

from .synthetic import SyntheticGenerator
from .fldigi import FldigiGenerator
from .wsjtx import WsjtxGenerator
from .analog import AnalogGenerator
from .sstv import SstvGenerator
from .packet import PacketGenerator
from .digivoice import DigivoiceGenerator
from .cw import CwCliGenerator
from .msk144 import Msk144Generator
from .minimodem import MinimodemGenerator
from .sameeas import SameeasGenerator
from .ardop import ArdopGenerator
from .js8call import Js8callGenerator
from .op25 import Op25Generator
from .hacktv import HacktvGenerator

GENERATORS = {
    "synthetic": SyntheticGenerator,
    "fldigi": FldigiGenerator,
    "wsjtx": WsjtxGenerator,
    "analog": AnalogGenerator,
    "sstv": SstvGenerator,
    "packet": PacketGenerator,
    "digivoice": DigivoiceGenerator,
    "cw": CwCliGenerator,
    "msk144": Msk144Generator,
    "minimodem": MinimodemGenerator,
    "sameeas": SameeasGenerator,
    "ardop": ArdopGenerator,
    "js8call": Js8callGenerator,
    "op25": Op25Generator,
    "hacktv": HacktvGenerator,
}

__all__ = ["GENERATORS"]
