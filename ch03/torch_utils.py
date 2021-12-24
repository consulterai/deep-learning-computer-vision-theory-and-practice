import torch
from ptflops import get_model_complexity_info
from torchvision.models import resnet18



def GET_FLOPS(model, shape):
    out_name = str(model.__class__) \
                   .split("\'>")[0].split('.')[-1] + "_flops.txt"
    # ost = sys.stdout
    ost = open(out_name, 'w')
    if len(shape) > 3:
        shape = (shape[1], shape[2], shape[3])
    macs, params = get_model_complexity_info(model, shape,
                                             as_strings=True,
                                             print_per_layer_stat
                                             =True,
                                             ost=ost)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs, params


if __name__ == '__main__':
    model = resnet18(pretrained=False)
    print(GET_FLOPS(model, (1, 3, 112, 112)))
    # out put is :
    # Computational complexity:       0.49 GMac
    # Number of parameters:           11.69 M
    # ('0.49 GMac', '11.69 M')
