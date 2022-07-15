# from detectron2.config import CfgNode
import copy

def get_cfg():
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return copy.deepcopy(_C)
