from lib.models.backbone.GoogleNet import GoogleNet
from lib.models.backbone.ResNet50Dilated import ResNet50Dilated
from lib.models.backbone.FBNet import FBNet
from lib.models.backbone.LightTrack import LightTrack
from lib.models.backbone.AlexNet import AlexNet
from lib.models.backbone.WaveMLP import WaveMLP


Backbones = {
    'ResNet50Dilated': ResNet50Dilated,
    'GoogleNet': GoogleNet,
    # 'MobileNetV2': MobileNetV2,
    'FBNet': FBNet,
    'LightTrack': LightTrack,
    'AlexNet': AlexNet,
    'WaveMLP': WaveMLP,
}
