import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        
        # Resource: conv1_module
        self.conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode='zeros'
        )
        
        # Resource: bn1_module
        self.bn1 = nn.BatchNorm2d(
            num_features=64,
            eps=1.0e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True
        )
        
        # Resource: shared_relu_module
        # Note: ReLU is a stateless module in PyTorch, but we instantiate it to respect resource identity.
        # inplace=True is set in the module constructor.
        self.relu = nn.ReLU(inplace=True)
        
        # Resource: conv2_module
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode='zeros'
        )
        
        # Resource: bn2_module
        self.bn2 = nn.BatchNorm2d(
            num_features=64,
            eps=1.0e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True
        )
        
        # Resource: post_add_gelu_module
        # approximate='none' is the default for nn.GELU
        self.activation = nn.GELU(approximate='none')

    def forward(self, x):
        # Dataflow: x -> main.input -> conv1.input
        # main.input is a boundary port, effectively just passing x to the first op in the 'main' branch.
        h = x
        
        # conv1
        h = self.conv1(h)
        
        # bn1
        h = self.bn1(h)
        
        # relu (in-place on input tensor h)
        # The YAML specifies target: input, which means the tensor arriving at the input port is mutated.
        # nn.ReLU(inplace=True) does exactly this.
        h = self.relu(h)
        
        # conv2
        h = self.conv2(h)
        
        # bn2
        h = self.bn2(h)
        
        # main.output is h
        
        # merge: add(main, shortcut) with inplace on main
        # main is h, shortcut is x
        # h.add_(x) mutates h in place
        h.add_(x)
        
        # activation
        y = self.activation(h)
        
        return y