"""
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
https://stackoverflow.com/questions/58362892/resnet-18-as-backbone-in-faster-r-cnn
"""
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


class FRCNNMobileNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # load a pre-trained model for classification and return
        # only the features
        # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        resnet_net = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        backbone = nn.Sequential(*modules)
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        # backbone.out_channels = 1280
        backbone.out_channels = 512

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )

        # put the pieces together inside a FasterRCNN model
        self.model = FasterRCNN(
            backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=100,
            max_size=300,
        )

    def forward(self, inp, labels=None):
        if labels is None:
            return self.model(inp)
        else:
            return self.model(inp, labels)
