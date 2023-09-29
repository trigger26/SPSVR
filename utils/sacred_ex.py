import os

from sacred import Ingredient

ing_base = Ingredient('base')
ing_train = Ingredient('train')
ing_test = Ingredient('test')


@ing_base.config
def base_cfg():
    # ==================== Dataset Config ====================
    dataset = ""
    num_class = 0
    label_dir = ""
    data_dir = ""

    # ==================== Device Config ====================
    gpus = []
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
    num_workers = 0
    batch_size = 0

    # ==================== Train / Query / Database Split Config ====================
    train_index = []
    query_index = []
    database_index = []

    # ==================== Model Config ====================
    code_length = 0
    n_frame = 0
    duration = 0


@ing_train.config
def train_cfg():
    epochs = 0
    warmup_epochs = 0
    lr = 0
    min_lr = 0
    weight_decay = 0
    optim = ""
    scheduler_gamma = 0
    print_freq = 0
    save_freq = 0
    checkpoint = None


@ing_test.config
def test_cfg():
    model_path = None


def parse_config(_config):
    """ Parse base configurations & add new configurations """
    config = {}
    config.update(_config['base'])
    config.update(_config['train'])
    config.update(_config['test'])
    config = Map({k: v for k, v in config.items()})

    # ==================== Path Config ====================
    data_dir = config['data_dir']
    video_names = sorted(os.listdir(data_dir))

    config['video_names'] = video_names

    # ==================== Train / Query / Database Split Config ====================
    config['train_names'] = [video_names[i] for i in config['train_index']]
    config['query_names'] = [video_names[i] for i in config['query_index']]
    config['database_names'] = [video_names[i] for i in config['database_index']]
    config['eval_names'] = sorted(config['query_names'] + config['database_names'])

    return config


class Map(dict):
    """ Support getting dict item by dot(.) operation """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, obj):
        new_dict = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_dict[k] = Map(v)
                else:
                    new_dict[k] = v
        else:
            raise TypeError(f"`obj` must be a dict, got {type(obj)}")
        super(Map, self).__init__(new_dict)
