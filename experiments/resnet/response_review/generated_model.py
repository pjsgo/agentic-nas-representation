import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        
        # Define modules based on resource mapping
        # conv1: conv1_module
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
        
        # bn1: bn1_module
        self.bn1 = nn.BatchNorm2d(
            num_features=64,
            eps=1.0e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True
        )
        
        # relu: shared_relu_module
        # Note: ReLU is stateless, but we instantiate it to respect the 'shared' resource ID concept
        # if it were a stateful module. Here it's just a functional operation wrapper.
        self.relu = nn.ReLU(inplace=True)
        
        # conv2: conv2_module
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
        
        # bn2: bn2_module
        self.bn2 = nn.BatchNorm2d(
            num_features=64,
            eps=1.0e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True
        )
        
        # activation: shared_relu_module
        # The YAML specifies 'shared_relu_module' for both 'relu' and 'activation'.
        # In PyTorch, nn.ReLU is stateless. Sharing the instance is semantically equivalent to using the same function.
        # We assign the same instance to reflect the resource mapping strictly.
        self.activation = self.relu

    def forward(self, x):
        # Dataflow: x -> main.input -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> main.output
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out) # inplace=True, target=input
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Dataflow: main.output -> merge.main, x -> merge.shortcut
        # merge: add, inplace=True, target=main
        # main.add_(shortcut) => out.add_(x)
        out.add_(x)
        
        # Dataflow: merge.output -> activation.input -> y
        out = self.activation(out) # inplace=True, target=input
        
        return out