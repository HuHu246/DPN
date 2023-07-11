import os
import torch
import yaml
import torch.nn as nn
import parser
from model import three_view_net

def moving_average(net1, net2, ema_decay=0.9999):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data = param1.data * (1.0 - ema_decay) + param2.data * ema_decay


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1 # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s'%dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
#---------------------------
def save_network(network, dirname, epoch_label):
    if not os.path.isdir('./model/'+dirname):
        os.mkdir('./model/'+dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth'% epoch_label
    else:
        save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',dirname,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
#  Load model for resume
#---------------------------
def load_network(name, opt):
    # Load config
    dirname = os.path.join('./model',name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    if not epoch=='last':
       epoch = int(epoch)
    config_path = os.path.join(dirname,'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    opt.name = config['name']
    opt.data_dir = config['data_dir']
    opt.train_all = config['train_all']
    opt.color_jitter = config['color_jitter']
    opt.batchsize = config['batchsize']
    opt.h = config['h']
    opt.w = config['w']
    opt.stride = config['stride']

    if 'pool' in config:
        opt.pool = config['pool']
    if 'h' in config:
        opt.h = config['h']
        opt.w = config['w']
    if 'gpu_ids' in config:
        opt.gpu_ids = config['gpu_ids']
    opt.lr = config['lr']
    opt.nclasses = config['nclasses']
    opt.block = config['block']

    model = three_view_net(opt.nclasses, block=opt.block)

    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth'% epoch
    else:
        save_filename = 'net_%s.pth'% epoch
    # save_filename = 'net_099.pth'
    save_path = os.path.join('./model',name,save_filename)
    print('Load the model from %s'%save_path)
    network = model
    network.load_state_dict(torch.load(save_path))
    return network, opt, epoch



def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for data1, data3 in zip(loader['satellite'], loader['drone']):
        input1 = data1[0].cuda()
        input3 = data3[0].cuda()

        b = input1.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input1, input3)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
