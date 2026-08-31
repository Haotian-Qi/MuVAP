import argparse
from os import makedirs
from os.path import dirname, exists, join
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

"""
Encoder should take a wave file, then return an embedding
"""


class CPC(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        # 16 kHz / 160 in the conv stack, halved twice by `downsample`.
        self.frame_hz = 25.0
        self.encoder = load_CPC(True)
        self.in_dim = self.encoder.gEncoder.conv4.out_channels
        self.out_dim = 256

        self.downsample = nn.Sequential(
            get_cnn_layer(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                kernel=[3],
                stride=[2],
                dilation=[1],
                activation="GELU",
            ),
            get_cnn_layer(
                in_dim=self.out_dim,
                out_dim=self.out_dim,
                kernel=[3],
                stride=[2],
                dilation=[1],
                activation="GELU",
            ),
        )

        self.freeze()

    def forward(self, waveform):
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)
        z = self.encoder.gEncoder(waveform)
        z = z.permute(0, 2, 1)
        z = self.encoder.gAR(z)
        z = self.downsample(z)
        return z

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)


"""

#############################################################
#############################################################
WARNING - ATTENTION - HEAR YEE HEAAAR YEEEE
#############################################################
#############################################################

Most of the code in this file are scaled down (and heavily copied) versions of

https://github.com/facebookresearch/CPC_audio

Please checkout their codebase if you are interested in CPC networks
----------------------------------------------------------------------------

torch.hub downloads to: `$HOME/.cache/torch/hub/checkpoints/`
Explicit checkpoint path saved manually in "assets/" see CHECKPOINTS below.
"""


def repo_root():
    """
    Returns the absolute path to the git repository
    """
    root = dirname(__file__)
    root = dirname(root)
    return root


CHECKPOINTS = {
    "cpc": join(repo_root(), "assets/checkpoints/cpc/60k_epoch4-d0f474de.pt")
}
NAMES = list(CHECKPOINTS.keys())


class ChannelNorm(nn.Module):
    """
    Most of the code in this file are scaled down (and heavily copied) versions of
        https://github.com/facebookresearch/CPC_audio
    """

    def __init__(self, numFeatures, epsilon=1e-05, affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):

        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean) * torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CPCEncoder(nn.Module):
    """
    Most of the code in this file are scaled down (and heavily copied) versions of
        https://github.com/facebookresearch/CPC_audio
    """

    def __init__(self, sizeHidden=512, normMode="layerNorm"):
        super(CPCEncoder, self).__init__()
        normLayer = ChannelNorm
        self.dimEncoded = sizeHidden
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160

    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        return x


class CPCAR(nn.Module):
    """
    Most of the code in this file are scaled down (and heavily copied) versions of
        https://github.com/facebookresearch/CPC_audio
    """

    def __init__(
        self, dimEncoded, dimOutput, keepHidden, nLevelsGRU, mode="GRU", reverse=False
    ):

        super(CPCAR, self).__init__()
        self.RESIDUAL_STD = 0.1

        if mode == "LSTM":
            self.baseNet = nn.LSTM(
                dimEncoded, dimOutput, num_layers=nLevelsGRU, batch_first=True
            )
        elif mode == "RNN":
            self.baseNet = nn.RNN(
                dimEncoded, dimOutput, num_layers=nLevelsGRU, batch_first=True
            )
        else:
            self.baseNet = nn.GRU(
                dimEncoded, dimOutput, num_layers=nLevelsGRU, batch_first=True
            )

        self.hidden = None
        self.keepHidden = keepHidden
        self.reverse = reverse

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        if self.reverse:
            x = torch.flip(x, [1])
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()

        # For better modularity, a sequence's order should be preserved
        # by each module
        if self.reverse:
            x = torch.flip(x, [1])
        return x


class CPCModel(nn.Module):
    """
    Most of the code in this file are scaled down (and heavily copied) versions of
        https://github.com/facebookresearch/CPC_audio
    """

    def __init__(self, encoder, AR):
        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(encodedData)
        return cFeature, encodedData, label


def load_CPC(load_state_dict=True):
    """
    Contrast predictive learning model for audio data
    pretrained: if True, load a model trained on libri-light 60k
    (https://arxiv.org/abs/1912.07875)
    **kwargs : see cpc/cpc_default_config to get the list of possible arguments

    Most of the code in this file are scaled down (and heavily copied) versions of
        https://github.com/facebookresearch/CPC_audio
    """

    def loadArgs(args, locArgs, forbiddenAttr=None):
        for k, v in vars(locArgs).items():
            if forbiddenAttr is not None:
                if k not in forbiddenAttr:
                    setattr(args, k, v)
            else:
                setattr(args, k, v)

    def get_default_cpc_config():
        parser = argparse.ArgumentParser()

        # Run parameters
        group = parser.add_argument_group(
            "Architecture configuration",
            description="The arguments defining the model's architecture.",
        )
        group.add_argument(
            "--hiddenEncoder",
            type=int,
            default=256,
            help="Hidden dimension of the encoder network.",
        )
        group.add_argument(
            "--hiddenGar",
            type=int,
            default=256,
            help="Hidden dimension of the auto-regressive network",
        )
        group.add_argument(
            "--nPredicts", type=int, default=12, help="Number of steps to predict."
        )
        group.add_argument(
            "--negativeSamplingExt",
            type=int,
            default=128,
            help="Number of negative samples to take.",
        )
        group.add_argument("--learningRate", type=float, default=2e-4)
        group.add_argument(
            "--schedulerStep",
            type=int,
            default=-1,
            help="Step of the learning rate scheduler: at each "
            "step the learning rate is divided by 2. Default: "
            "no scheduler.",
        )
        group.add_argument(
            "--schedulerRamp",
            type=int,
            default=None,
            help="Enable a warm up phase for the learning rate: "
            "adds a linear ramp of the given size.",
        )
        group.add_argument(
            "--beta1",
            type=float,
            default=0.9,
            help="Value of beta1 for the Adam optimizer",
        )
        group.add_argument(
            "--beta2",
            type=float,
            default=0.999,
            help="Value of beta2 for the Adam optimizer",
        )
        group.add_argument(
            "--epsilon",
            type=float,
            default=1e-08,
            help="Value of epsilon for the Adam optimizer",
        )
        group.add_argument(
            "--sizeWindow",
            type=int,
            default=20480,
            help="Number of frames to consider at each batch.",
        )
        group.add_argument(
            "--nEpoch", type=int, default=200, help="Number of epoch to run"
        )
        group.add_argument(
            "--samplingType",
            type=str,
            default="samespeaker",
            choices=["samespeaker", "uniform", "samesequence", "sequential"],
            help="How to sample the negative examples in the CPC loss.",
        )
        group.add_argument(
            "--nLevelsPhone",
            type=int,
            default=1,
            help="(Supervised mode only). Number of layers in "
            "the phone classification network.",
        )
        group.add_argument(
            "--cpc_mode",
            type=str,
            default=None,
            choices=["reverse", "none"],
            help="Some variations on CPC.",
        )
        group.add_argument(
            "--encoder_type",
            type=str,
            choices=["cpc", "mfcc", "lfb"],
            default="cpc",
            help="Replace the encoder network by mfcc features or learned filter banks",
        )
        group.add_argument(
            "--normMode",
            type=str,
            default="layerNorm",
            choices=["instanceNorm", "ID", "layerNorm", "batchNorm"],
            help="Type of normalization to use in the encoder "
            "network (default is layerNorm).",
        )
        group.add_argument(
            "--onEncoder",
            action="store_true",
            help="(Supervised mode only) Perform the "
            "classification on the encoder's output.",
        )
        group.add_argument(
            "--random_seed", type=int, default=None, help="Set a specific random seed."
        )
        group.add_argument(
            "--speakerEmbedding",
            type=int,
            default=0,
            help="(Depreciated) Feed the prediction network with "
            "speaker embeddings along with the usual sequence.",
        )
        group.add_argument(
            "--arMode",
            default="LSTM",
            choices=["GRU", "LSTM", "RNN", "no_ar", "transformer"],
            help="Architecture to use for the auto-regressive "
            "network (default is lstm).",
        )
        group.add_argument(
            "--nLevelsGRU",
            type=int,
            default=1,
            help="Number of layers in the autoregressive network.",
        )
        group.add_argument(
            "--rnnMode",
            type=str,
            default="transformer",
            choices=[
                "transformer",
                "RNN",
                "LSTM",
                "linear",
                "ffd",
                "conv4",
                "conv8",
                "conv12",
            ],
            help="Architecture to use for the prediction network",
        )
        group.add_argument(
            "--dropout",
            action="store_true",
            help="Add a dropout layer at the output of the prediction network.",
        )
        group.add_argument(
            "--abspos",
            action="store_true",
            help="If the prediction network is a transformer, "
            "active to use absolute coordinates.",
        )
        return parser.parse_args([])

    # from cpc.model import CPCModel as cpcmodel
    # from cpc.cpc_default_config import get_default_cpc_config
    # from cpc.feature_loader import getEncoder, getAR, loadArgs
    # from cpc.feature_loader import loadArgs

    locArgs = get_default_cpc_config()
    if exists(CHECKPOINTS["cpc"]):
        checkpoint = torch.load(
            CHECKPOINTS["cpc"], map_location="cpu", weights_only=True
        )
    else:
        checkpoint_url = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt"
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_url, progress=False, map_location="cpu"
        )
        makedirs(dirname(CHECKPOINTS["cpc"]), exist_ok=True)
        torch.save(checkpoint, CHECKPOINTS["cpc"])
    loadArgs(locArgs, argparse.Namespace(**checkpoint["config"]))
    # encoderNet = getEncoder(locArgs)
    encoderNet = CPCEncoder(locArgs.hiddenEncoder, locArgs.normMode)
    # arNet = getAR(locArgs)
    arNet = CPCAR(
        locArgs.hiddenEncoder,
        locArgs.hiddenGar,
        locArgs.samplingType == "sequential",
        locArgs.nLevelsGRU,
        mode=locArgs.arMode,
        reverse=locArgs.cpc_mode == "reverse",
    )
    # model = cpcmodel(encoderNet, arNet)
    model = CPCModel(encoderNet, arNet)

    # always load pretrained
    if load_state_dict:
        model.load_state_dict(checkpoint["weights"], strict=False)
    model.name = "cpc"
    return model


class LayerNorm(nn.Module):
    def __init__(self, dim: int, rearrange_outputs: bool = True) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.in_rearrange = Rearrange("b d t -> b t d")
        if rearrange_outputs:
            self.out_rearrange = Rearrange("b t d -> b d t")
        else:
            self.out_rearrange = nn.Identity()

    def __repr__(self):
        return str(self.ln)

    def forward(self, x):
        return self.out_rearrange(self.ln(self.in_rearrange(x)))


class CConv1d(nn.Conv1d):
    """source: https://github.com/pytorch/pytorch/issues/1333"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        padding_value=0,
        bias=True,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )

        ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        pad_dim1_pre = ks - 1
        pad_dim1_post = 0
        if dilation > 0:
            pad_dim1_pre *= dilation
        pad = (pad_dim1_pre, pad_dim1_post)
        self.pad = nn.ConstantPad1d(padding=pad, value=padding_value)

    def debug_weights(self, type="sum"):
        w = 1.0
        if type == "mean":
            w = 1.0 / self.kernel_size[0]

        elif type == "range":
            k = self.kernel_size[0]
            w = torch.arange(1, k + 1).float().pow(2)
            w = w.repeat(self.out_channels, self.in_channels, 1)
            self.weight.data = w
            if self.bias:
                self.bias.data = self.bias.data.fill_(0.0)
            return None

        self.weight.data = self.weight.data.fill_(w)
        if self.bias:
            self.bias.data = self.bias.data.fill_(0.0)

    def forward(self, input):
        return super().forward(self.pad(input))


def get_cnn_layer(
    in_dim: int,
    out_dim: int,
    kernel: List[int] = [3],
    stride: List[int] = [2],
    dilation: List[int] = [1],
    padding: List[int] = [0],
    activation: str = "GELU",
):
    layers = []
    layers.append(Rearrange("b t d -> b d t"))
    for k, s, d, p in zip(kernel, stride, dilation, padding):
        layers.append(
            CConv1d(in_dim, out_dim, kernel_size=k, stride=s, dilation=d, padding=p)
        )
        layers.append(LayerNorm(out_dim))
        if activation is not None:
            layers.append(getattr(torch.nn, activation)())
    layers.append(Rearrange("b d t -> b t d"))
    return nn.Sequential(*layers)
