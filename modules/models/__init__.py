from .tree_model import DummyTreeModel
from .dummy_const_model import DummyConstModel
from .dummy_rand_model import DummyRandModel

DICT_MODELS = {
    "tree": DummyTreeModel,
    "dummy": DummyConstModel,
    "rand": DummyRandModel,
}
