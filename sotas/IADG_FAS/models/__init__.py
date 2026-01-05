try:
    from .framework import FeatEmbedder, FeatExtractor, DepthEstmator, Framework
    from .base_block import Conv_block_gate, conv3x3, Basic_block_gate
    from . import dkg_module
except ImportError:
    from models import FeatEmbedder, FeatExtractor, DepthEstmator, Framework
    from models import Conv_block_gate, conv3x3, Basic_block_gate
    from models import dkg_module
