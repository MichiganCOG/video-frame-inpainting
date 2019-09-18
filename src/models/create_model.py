import json
import os
import sys
from warnings import warn

from .tai.tai import TAIFillInModel
from .optical_flow_fill_in.OFFillInModel import OFFillInModel
from .self_attention.self_attention import SCTSkipConScaledTForwardFillInModel, SCTSkipConScaledTInwardFillInModel, \
    SCTSkipConScaledTRandomFillInModel, SCTSkipConScaledTRandomBFillInModel, SCTSkipConScaledTRandomCFillInModel, \
    SCTSkipConScaledTRandomDFillInModel, SCTBypassScaledTForwardFillInModel, \
    SCTFrameEncDecBNSkipConScaledTForwardFillInModel
from .mcnet.mcnet import MCNetFillInModel
from .slomo.slomo import SloMoFillInModel
from .twi.twi import TimeWeightedInterpolationFillInModel
from .bi_twa.bi_twa import BidirectionalTimeWeightedAverageFillInModel
from .bi_sa.bi_sa import BidirectionalSimpleAverageFillInModel
from .tw_p_f.tw_p_f import TimeWeightedPFFillInModel

def create_model(model_key):
    """Produce a generator or a discriminator identified by the given model key.

    TODO: Create a dictionary with model_key - model pairs inside, then just retrieve models from that dictionary

    :param model_key: The key used to identify the model to return
    """

    if model_key == 'TAI_gray':
        return TAIFillInModel(64, 1, 3, 51, num_block=5)
    if model_key == 'TAI_color':
        return TAIFillInModel(64, 3, 3, 51, num_block=4)
    if model_key == 'OFFillInModel':
        return OFFillInModel()
    if model_key == 'MCNet_gray':
        return MCNetFillInModel(64, 1, 3)
    if model_key == 'MCNet_color':
        return MCNetFillInModel(64, 3, 3)
    if model_key == 'SCTSkipConScaledTForward_gray':
        return SCTSkipConScaledTForwardFillInModel(1, 2, 3, 256, 2048)
    if model_key == 'SCTSkipConScaledTForward_color':
        return SCTSkipConScaledTForwardFillInModel(3, 2, 3, 256, 2048)
    if model_key == 'SCTSkipConScaledTInward_gray':
        return SCTSkipConScaledTInwardFillInModel(1, 2, 3, 256, 2048)
    if model_key == 'SCTSkipConScaledTInward_color':
        return SCTSkipConScaledTInwardFillInModel(3, 2, 3, 256, 2048)
    if model_key == 'SCTSkipConScaledTRandom_gray':
        return SCTSkipConScaledTRandomFillInModel(1, 2, 3, 256, 2048)
    if model_key == 'SCTSkipConScaledTRandom_color':
        return SCTSkipConScaledTRandomFillInModel(3, 2, 3, 256, 2048)
    if model_key == 'SCTSkipConScaledTRandomB_gray':
        return SCTSkipConScaledTRandomBFillInModel(1, 2, 3, 256, 2048)
    if model_key == 'SCTSkipConScaledTRandomB_color':
        return SCTSkipConScaledTRandomBFillInModel(3, 2, 3, 256, 2048)
    if model_key == 'SCTSkipConScaledTRandomC_gray':
        return SCTSkipConScaledTRandomCFillInModel(1, 2, 3, 256, 2048)
    if model_key == 'SCTSkipConScaledTRandomC_color':
        return SCTSkipConScaledTRandomCFillInModel(3, 2, 3, 256, 2048)
    if model_key == 'SCTSkipConScaledTRandomD_gray':
        return SCTSkipConScaledTRandomDFillInModel(1, 2, 3, 256, 2048)
    if model_key == 'SCTSkipConScaledTRandomD_color':
        return SCTSkipConScaledTRandomDFillInModel(3, 2, 3, 256, 2048)
    if model_key == 'SCTBypassScaledTForward_gray':
        return SCTBypassScaledTForwardFillInModel(1, 2, 3, 256, 2048)
    if model_key == 'SCTBypassScaledTForward_color':
        return SCTBypassScaledTForwardFillInModel(3, 2, 3, 256, 2048)
    if model_key == 'SCTFrameEncDecBNSkipConScaledTForward_gray':
        return SCTFrameEncDecBNSkipConScaledTForwardFillInModel(1, 2, 3, 256, 2048)
    if model_key == 'SCTFrameEncDecBNSkipConScaledTForward_color':
        return SCTFrameEncDecBNSkipConScaledTForwardFillInModel(3, 2, 3, 256, 2048)
    if model_key == 'SloMoFillInModel_color':
        return SloMoFillInModel(32, 3)
    if model_key == 'SloMoFillInModel_gray':
        return SloMoFillInModel(32, 1)
    if model_key == 'TimeWeightedInterpolationFillInModel_gray':
        return TimeWeightedInterpolationFillInModel(64, 1, 3, 51, num_block=5)
    if model_key == 'TimeWeightedInterpolationFillInModel_color':
        return TimeWeightedInterpolationFillInModel(64, 3, 3, 51, num_block=4)
    if model_key == 'BidirectionalSimpleAverageFillInModel_gray':
        return BidirectionalSimpleAverageFillInModel(64, 1, 3)
    if model_key == 'BidirectionalSimpleAverageFillInModel_color':
        return BidirectionalSimpleAverageFillInModel(64, 3, 3)
    if model_key == 'BidirectionalTimeWeightedAverageFillInModel_gray':
        return BidirectionalTimeWeightedAverageFillInModel(64, 1, 3)
    if model_key == 'BidirectionalTimeWeightedAverageFillInModel_color':
        return BidirectionalTimeWeightedAverageFillInModel(64, 3, 3)
    if model_key == 'TimeWeightedPFFillInModel':
        return TimeWeightedPFFillInModel()

    print('Could not determine the model to create from key %s. Treating key as file path...' % model_key)
    if os.path.isfile(model_key):
        with open(model_key, 'r') as f:
            # Load the JSON file to a dict
            model_info = json.load(f)
            return _construct_model_from_dict(model_info)

    print('Could not find file %s. Treating key as JSON string...' % model_key)
    try:
        model_info = json.loads(model_key)
    except ValueError:
        raise RuntimeError('Failed to parse model key as a JSON object')

    return _construct_model_from_dict(model_info)


def _construct_model_from_dict(model_info):
    # Check that correct fields are specified in the file, and that they're the correct object types
    assert(isinstance(model_info.get('class'), str) or isinstance(model_info.get('class'), unicode))
    assert(isinstance(model_info.get('args'), list))
    assert(isinstance(model_info.get('kwargs'), dict))
    # Construct the architecture
    model_class = getattr(sys.modules[__name__], model_info['class'])
    return model_class(*model_info['args'], **model_info['kwargs'])