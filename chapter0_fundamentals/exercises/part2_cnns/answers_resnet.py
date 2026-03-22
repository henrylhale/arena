import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm


# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part2_cnns.tests as tests
import part2_cnns.utils as utils
from plotly_utils import line
import plotly.express as px


class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(x, t.zeros(x.shape, device=x.device))


tests.test_relu(ReLU)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        rw = 1 / np.sqrt(in_features)
        w = t.zeros((out_features, in_features))
        self.weight = nn.Parameter(w.uniform_(-1 * rw, rw))
        if bias:
            b = t.zeros((out_features))
            rb = 1 / np.sqrt(in_features)
            self.bias = nn.Parameter(b.uniform_(-1 * rb, rb))
        else:
            self.bias = None


    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        x = einops.einsum(self.weight, x, "out in, ... in -> ... out") 
        if self.bias is not None:
            x += self.bias
        return x

    def extra_repr(self) -> str:
        print(f"{in_features} to {out_features} linear layer, bias={self.bias is not None}")



tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1], device=input.device)).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])




device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
# device = "cpu"

# If this is CPU, we recommend figuring out how to get cuda access (or MPS if you're on a Mac).
print(device)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.

        We assume kernel is square, with height = width = `kernel_size`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        r = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        w = t.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.weight = nn.Parameter(w.uniform_(-1 * r, r))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d, which you can import."""
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


tests.test_conv2d_module(Conv2d)
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """Call the functional version of maxpool2d."""
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])


class Sequential(nn.Module):
    _modules: dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules)  # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules)  # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: Tensor) -> Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            x = mod(x)
        return x



class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, " num_features"]
    running_var: Float[Tensor, " num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """

        if self.training:
            m = t.mean(x, dim=[0,2,3], keepdim=True)
            v = t.var(x, dim=[0,2,3], keepdim=True, unbiased=False)
            v_unbiased = t.var(x, dim=[0,2,3], keepdim=False, unbiased=True)
            w = einops.repeat(self.weight, "n -> 1 n 1 1")
            b = einops.repeat(self.bias, "n -> 1 n 1 1")

            x = ((x - m) / t.sqrt(v + self.eps)) * w + b

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * m[0,:,0,0] 
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * v_unbiased
            self.num_batches_tracked += 1
        else:
            m = einops.repeat(self.running_mean, "n -> 1 n 1 1")
            v = einops.repeat(self.running_var, "n -> 1 n 1 1")
            w = einops.repeat(self.weight, "n -> 1 n 1 1")
            b = einops.repeat(self.bias, "n -> 1 n 1 1")

            x = ((x - m) / t.sqrt(v + self.eps)) * w + b
        return x

    def extra_repr(self) -> str:
        print(f"BatchNorm2d")


tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)



class AveragePool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return t.mean(x, dim=(2, 3))



class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a
        `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right
        branch. Declare it second using another `Sequential`.
        """
        super().__init__()
        is_shape_preserving = (first_stride == 1) and (in_feats == out_feats)  # determines if right branch is identity

        self.left_block = Sequential(
            Conv2d(in_channels=in_feats, out_channels=out_feats, padding=1, kernel_size=3, stride=first_stride),
            BatchNorm2d(num_features=out_feats),
            ReLU(),
            Conv2d(in_channels=out_feats, kernel_size=3, padding=1, out_channels=out_feats),
            BatchNorm2d(num_features=out_feats),
        )

        if is_shape_preserving:
            self.right_block = nn.Identity()
        else:
            self.right_block = Sequential(
                Conv2d(in_channels=in_feats, out_channels=out_feats, kernel_size=1, padding=0, stride=first_stride),
                BatchNorm2d(num_features=out_feats),
            )

        self.relu = ReLU()


    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass. If no downsampling block is present, the addition should just add
        the left branch's output to the input.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)
        """
        return self.relu(self.left_block(x) + self.right_block(x))


tests.test_residual_block(ResidualBlock)



class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """
        An n_blocks-long sequence of ResidualBlock where only the first block uses the provided
        stride.
        """
        super().__init__()
        
        self.block = nn.Sequential(
            ResidualBlock(in_feats=in_feats, out_feats=out_feats, first_stride=first_stride),
            *[ResidualBlock(in_feats=out_feats, out_feats=out_feats) for i in range(1, n_blocks)]
        )


    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.block(x)


tests.test_block_group(BlockGroup)



class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        out_feats0 = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        feature_numbers = [out_feats0] + out_features_per_group

        # YOUR CODE HERE - define all components of resnet34
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(num_features=64),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Sequential(
                *[BlockGroup(
                    n_blocks_per_group[i],
                    feature_numbers[i],
                    feature_numbers[i + 1],
                    first_strides_per_group[i]) for i in range(4)
                 ]
            ),
            AveragePool(),
            Linear(in_features=512, out_features=1000),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        """
        # for i, layer in enumerate(self.model._modules.values()):
          # print(f"running layer {i}: {type(layer).__name__}", flush=True)
          # x = layer(x)
        # return x

        return self.model(x)


my_resnet = ResNet34()


# (1) Test via helper function `print_param_count`
from tabulate import tabulate

target_resnet = models.resnet34()  # without supplying a `weights` argument, we just initialize with random weights
# print(tabulate(utils.print_param_count(my_resnet, target_resnet, display_df=False), headers='keys', tablefmt='pretty'))
print(tabulate(utils.print_param_count(my_resnet, target_resnet, display_df=False), headers='keys', tablefmt='pretty'))




def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
    """Copy over the weights of `pretrained_resnet` to your resnet."""

    # Get the state dictionaries for each model, check they have the same number of parameters &
    # buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the
    # pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet


pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
my_resnet = copy_weights(my_resnet, pretrained_resnet).to(device)
print("Weights copied successfully!")




IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = section_dir / "resnet_inputs"

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]



IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

prepared_images = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0).to(device)
assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)


print(t.cuda.memory_allocated() / 1e9, "GB allocated")
print(t.cuda.memory_reserved() / 1e9, "GB reserved")


t.backends.cudnn.enabled = False

@t.inference_mode()
def predict(
    model: nn.Module, images: Float[Tensor, "batch rgb h w"]
) -> tuple[Float[Tensor, " batch"], Int[Tensor, " batch"]]:
    """
    Returns the maximum probability and predicted class for each image, as a tensor of floats and
    ints respectively.
    """
    print(f"setting model to eval mode", flush=True)
    model.eval()
    print("starting inference", flush=True)
    logits = model(images)
    print("done with inference", flush=True)
    return t.max(t.softmax(logits, dim=-1), dim=-1)


with open(section_dir / "imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())

# Check your predictions match those of the pretrained model
my_probs, my_predictions = predict(my_resnet, prepared_images)
pretrained_probs, pretrained_predictions = predict(pretrained_resnet, prepared_images)
assert (my_predictions == pretrained_predictions).all()
t.testing.assert_close(my_probs, pretrained_probs, atol=5e-4, rtol=0)  # tolerance of 0.05%
print("All predictions match!")


# Print out your predictions, next to the corresponding images
for i, img in enumerate(images):
    table = Table("Model", "Prediction", "Probability")
    table.add_row("My ResNet", imagenet_labels[my_predictions[i]], f"{my_probs[i]:.3%}")
    table.add_row(
        "Reference Model",
        imagenet_labels[pretrained_predictions[i]],
        f"{pretrained_probs[i]:.3%}",
    )
    rprint(table)
    fig = px.imshow(np.array(img))
    # fig.show(renderer="browser")


#conv2d sanity

x = t.randn(1, 3, 224, 224, device=device)
w = t.randn(64, 3, 7, 7, device=device)
out = F.conv2d(x, w, stride=2, padding=3)
t.cuda.synchronize()
print("F.conv2d works:", out.shape)

with t.inference_mode():
  out2 = pretrained_resnet(prepared_images)
print("Pretrained works:", out2.shape)


