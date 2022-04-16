
from model import losses

def build(loss_config):

    criterions = {}
  
    criterions["metric_loss"] =losses.MetricLoss(configs=loss_config.metric_loss,)

    criterions["pose_loss"] = losses.PointAlignmentLoss(loss_weight=1)
    
    return criterions
