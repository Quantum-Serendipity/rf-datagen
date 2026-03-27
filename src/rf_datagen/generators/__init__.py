"""Generator registry."""

from .synthetic import SyntheticGenerator
from .fldigi import FldigiGenerator
from .wsjtx import WsjtxGenerator
from .analog import AnalogGenerator
from .sstv import SstvGenerator
from .packet import PacketGenerator
from .digivoice import DigivoiceGenerator

GENERATORS = {
    "synthetic": SyntheticGenerator,
    "fldigi": FldigiGenerator,
    "wsjtx": WsjtxGenerator,
    "analog": AnalogGenerator,
    "sstv": SstvGenerator,
    "packet": PacketGenerator,
    "digivoice": DigivoiceGenerator,
}

__all__ = ["GENERATORS"]
