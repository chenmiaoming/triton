"""Resolve append-only collector revisions; original protocols stay immutable."""
from experiments.tma_reduction_layout.phase11 import common as c


def load_b():
    root = c.OUT/'stage_b/preregistration'
    choices = list((root/'revisions').glob('*/protocol.json'))
    path = max(choices,key=lambda p:int(p.parent.name)) if choices else root/'protocol.json'
    return c.read(path),path
