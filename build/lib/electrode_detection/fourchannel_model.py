import torch 

from ultralytics import YOLO
from ultralytics.models.yolo.obb.train import OBBTrainer

class FourChannelsOBBTrainer(OBBTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):

        model = super().get_model(
            cfg=cfg,
            weights=weights,
            verbose=verbose
        )

        if weights == None:
            return model

        first_conv = model.model[0].conv # in_channels = 4

        pretrained_conv = weights.model[0].conv # in_channels = 3

        # transfer RGB channel weights and init fourth chanel to 0
        with torch.no_grad():

            first_conv.weight[:, :3].copy_(pretrained_conv.weight)
            first_conv.weight[:, 3].zero_()

            if pretrained_conv.bias is not None:
                first_conv.bias.copy_(pretrained_conv.bias)

        return model

class FourChannelsOBBYolo(YOLO):

    def train(self, **kwargs):

        return super().train(trainer=FourChannelsOBBTrainer, **kwargs)