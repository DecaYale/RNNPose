from builder import losses_builder
from model.RNNPose import get_posenet_class
import model.RNNPose


def build(model_cfg,
          measure_time=False, testing=False):
    """build second pytorch instance.
    """

    criterions=losses_builder.build(model_cfg.loss)

    net = get_posenet_class(model_cfg.network_class_name)(
        criterions=criterions,
        opt=model_cfg)
    return net
