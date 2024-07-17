import os
import os.path as osp
import logging
from collections import OrderedDict
import json
from datetime import datetime


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def parse(args):
    phase = args.phase
    opt_path = args.config
    gpu_ids = args.gpu_ids
    enable_wandb = args.enable_wandb
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    #解析json格式的字符串 以顺序字典的形式存储在变量opt中
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # set log directory
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    experiments_root = os.path.join(
        'experiments', '{}_{}'.format(opt['name'], get_timestamp()))
    opt['path']['experiments_root'] = experiments_root
    #将实验结果的地址添加到opt中
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    # change dataset length limit
    opt['phase'] = phase

    # export CUDA_VISIBLE_DEVICES
    if gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
        gpu_list = gpu_ids
    else:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    if len(gpu_list) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    # debug
    if 'debug' in opt['name']:
        opt['train']['val_freq'] = 2
        opt['train']['print_freq'] = 2
        opt['train']['save_checkpoint_freq'] = 3
        opt['datasets']['train']['batch_size'] = 2
        opt['model']['beta_schedule']['train']['n_timestep'] = 10
        opt['model']['beta_schedule']['val']['n_timestep'] = 10
        opt['datasets']['train']['data_len'] = 6
        opt['datasets']['val']['data_len'] = 3

    # validation in train phase
    if phase == 'train':
        opt['datasets']['val']['data_len'] = 3

    # W&B Logging
    try:
        log_wandb_ckpt = args.log_wandb_ckpt
        opt['log_wandb_ckpt'] = log_wandb_ckpt
    except:
        pass
    try:
        log_eval = args.log_eval
        opt['log_eval'] = log_eval
    except:
        pass
    try:
        log_infer = args.log_infer
        opt['log_infer'] = log_infer
    except:
        pass
    opt['enable_wandb'] = enable_wandb
    
    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
# 将嵌套的字典和列表中的每个元素都转换为NoneDict或其相应的类型
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

#设置日志记录器
def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)#创建一个名为logger_name的日志记录器
    #创建了一个日志格式化器Formatter对象，用于定义日志信息的格式
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, '{}.log'.format(phase))
    #创建了一个FileHandler对象fh 将日志信息写入到指定的文件
    fh = logging.FileHandler(log_file, mode='w')
    #将之前定义的格式化器formatter应用到FileHandler对象fh上
    fh.setFormatter(formatter)
    #设置日志记录器的级别
    l.setLevel(level)
    #将FileHandler对象fh添加到日志记录器l中
    l.addHandler(fh)
    if screen:
        #创建了一个StreamHandler对象sh 将日志信息输出到屏幕上
        sh = logging.StreamHandler()
        #将之前定义的格式化器formatter应用到StreamHandler对象sh上
        sh.setFormatter(formatter)
        #将StreamHandler对象sh添加到日志记录器l中
        l.addHandler(sh)
