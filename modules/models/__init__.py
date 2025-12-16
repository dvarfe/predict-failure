from .tree_model import DummyTreeModel
from .dummy_const_model import DummyConstModel
from .dummy_rand_model import DummyRandModel
from .Cox import CoxTimeVaryingEstimator
from .NN_models.SP_model import SurvPredictor

DICT_MODELS = {
    "dummy_const": DummyConstModel,
    "dummy_rand": DummyRandModel,
    "tree": DummyTreeModel,
    "cox": CoxTimeVaryingEstimator,
    "sp": SurvPredictor,
}

MODEL_PARAMETERS = {
    'dummy_const': {},

    'dummy_rand': {
    },

    'tree': {},

    'cox': {
        'penalizer': {
            'type': 'float',
            'default': 0.0,
            'description': 'Коэффициент регуляризации для предотвращения переобучения',
            'min': 0.0,
            'max': 100000.0,
            'step': 0.001
        },
        'l1_ratio': {
            'type': 'float',
            'default': 0.0,
            'description': 'Соотношение L1 и L2 регуляризации (0 - только L2, 1 - только L1)',
            'min': 0.0,
            'max': 100000.0,
            'step': 0.01
        }
    },

    'sp': {
        'hidden_dim': {
            'type': 'int',
            'default': 64,
            'description': 'Количество скрытых нейронов в сети',
            'min': 8,
            'max': 1024,
            'step': 8
        },
        'lr': {
            'type': 'float',
            'default': 0.001,
            'description': 'Скорость обучения (learning rate)',
            'min': 0.0001,
            'max': 0.1,
            'step': 0.0001
        },
        'epochs': {
            'type': 'int',
            'default': 100,
            'description': 'Количество эпох обучения',
            'min': 1,
            'max': 1000,
            'step': 1
        }
    }
}
