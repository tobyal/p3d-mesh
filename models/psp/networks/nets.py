import torch.nn as nn
from models.psp.networks.decoder import Decoder
from models.psp.networks.encoder import PSP_PointTransformer


    
class PSPNet(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.encoder = PSP_PointTransformer(cfgs.encoder)
        self.decoder = Decoder(cfgs.decoder)
    
    def forward(self, pos, patch_center, patch_radius):
        feat = self.encoder(pos)
        return self.decoder(pos, feat, patch_center, patch_radius)
    

