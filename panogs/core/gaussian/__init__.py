"""3D Gaussian representation, models, and initialization."""
from panogs.core.gaussian.model import GaussianModel


def initialize_from_pointcloud(*args, **kwargs):
    from panogs.core.gaussian.initialization import initialize_from_pointcloud as _init
    return _init(*args, **kwargs)


__all__ = ["GaussianModel", "initialize_from_pointcloud"]
