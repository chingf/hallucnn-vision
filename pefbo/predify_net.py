import predify
from timm.models import efficientnet_b0

net = efficientnet_b0(pretrained=True)

predify.predify(net, f'pconfigs.toml')
