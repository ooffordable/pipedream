# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
#from .stage0 import Stage0
#from .stage1 import Stage1
#from .stage2 import Stage2

from stage0 import Stage0
from stage1 import Stage1
from stage2 import Stage2


class ResNet50Split(torch.nn.Module):
    def __init__(self):
        super(ResNet50Split, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self._initialize_weights()

    def forward(self, input0):
        (out0, out1) = self.stage0(input0)
        (out3, out2) = self.stage1(out0, out1)
        out4 = self.stage2(out3, out2)
        return out4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


def check_memory():
    model = ResNet50Split()
    input0 = torch.autograd.Variable(torch.rand((17, 3, 299, 299))).type(torch.FloatTensor).cuda(3)    
    

    
    
    stage = model.stage0.cuda(3)
    before = torch.cuda.memory_allocated()
    (out0, out1) = stage(input0)
    after = torch.cuda.memory_allocated()
    latent_size = after-before
    param_size = sum(p.storage().size() * p.storage().element_size()
                         for p in stage.parameters())
    print("stage param_size = %d, latent_size = %d, sum = %d" % (param_size>>6, latent_size>>6, (param_size+latent_size) >>6))

    stage = model.stage1.cuda(3)
    before = torch.cuda.memory_allocated()
    (out3, out2) = stage(out0, out1)
    after = torch.cuda.memory_allocated()
    latent_size = after-before
    param_size = sum(p.storage().size() * p.storage().element_size()
                         for p in stage.parameters())
    print("stage param_size = %d, latent_size = %d, sum = %d" % (param_size>>6, latent_size>>6, (param_size+latent_size) >>6))


    stage = model.stage2.cuda(3)
    before = torch.cuda.memory_allocated()
    out4 = stage(out3, out2)
    after = torch.cuda.memory_allocated()
    latent_size = after-before
    param_size = sum(p.storage().size() * p.storage().element_size()
                         for p in stage.parameters())
    print("stage param_size = %d, latent_size = %d, sum = %d" % (param_size>>6, latent_size>>6, (param_size+latent_size) >>6))

if __name__ == "__main__":
    check_memory()
