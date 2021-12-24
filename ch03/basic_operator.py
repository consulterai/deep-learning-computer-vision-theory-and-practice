from ch03.torch_utils import GET_FLOPS


def base_conv():
    import torch
    import torch.nn as nn
    base_conv = nn.Conv2d(in_channels=3, out_channels=2,
                          kernel_size=(3, 3), stride=(1, 1),
                          padding=(0, 0), bias=True,
                          padding_mode='zeros')
    base_conv.weight.data = torch.ones([2, 3, 3, 3], dtype=torch.float32)
    base_conv.bias.data = torch.tensor([4, 5], dtype=torch.float32)
    input = torch.ones([1, 3, 6, 6], dtype=torch.float32)
    out = base_conv(input)
    print(out)
    # Params: channel_o * channel_in * k1 * k2 + channel_o (如果有bias)
    # FLOPs:  channel_o * channel_in * k1 * k2 + channel_o (如果有bias)

    base_conv_flopcnt = nn.Conv2d(in_channels=18, out_channels=36,
                                  kernel_size=(3, 3), stride=(1, 1),
                                  padding=(0, 0), bias=True,
                                  padding_mode='zeros',
                                  groups=3)
    print(GET_FLOPS(base_conv_flopcnt, [1, 18, 112, 112]))


if __name__ == '__main__':
    base_conv()
