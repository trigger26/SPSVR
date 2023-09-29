from dataset import get_dataloader
from model import SPHashNet
import transforms
from utils.sacred_ex import ing_base, ing_train, ing_test, parse_config

from sacred import Experiment
from sacred.observers import FileStorageObserver
import argparse
import warnings
import numpy as np

import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cfg", "--config", dest="config", type=str)
args = parser.parse_args()

dataset = args.config.split('/')[-1].split('.')[0]
ex_name = 'code_generation'
ex = Experiment(ex_name, save_git_info=False, ingredients=[ing_base, ing_train, ing_test])
ex.observers.append(FileStorageObserver(f'log/sacred/{ex_name}/{dataset}'))


@ex.main
def main(_config):
    config = parse_config(_config)
    torch.backends.cudnn.benchmark = True

    # ==================== Dataloader ====================
    tfs = nn.Sequential(
        transforms.GroupResize((224, 224)),
        transforms.GroupConvertImageDtype(torch.float),
        transforms.GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    dataloader = get_dataloader(
        name=config.dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        data_dir=config.data_dir,
        label_dir=config.label_dir,
        num_class=config.num_class,
        video_names=config.eval_names,
        n_frame=config.n_frame,
        duration=config.duration,
        group_tfs=tfs
    )
    print(f'query & database: {len(dataloader.dataset)} samples, {len(dataloader)} batches.')

    # ==================== Model ====================
    net = SPHashNet(code_length=config.code_length, n_frame=config.n_frame)
    point_dict = torch.load(config.model_path, map_location=torch.device('cpu'))
    net.load_state_dict(point_dict['state_dict'])
    model = nn.DataParallel(net).cuda()
    print(f"Load checkpoint from {config.model_path}, epoch: {point_dict['epoch']}")
    ex.add_artifact(config.model_path)

    # ==================== Generate ====================
    generate(model, dataloader, config, ex)
    print("Finish generating hash codes.")


def generate(model, dataloader, config, ex):
    bins_video = {v: [] for v in config.eval_names}
    yps_video = {v: [] for v in config.eval_names}
    frame_ids_video = {v: [] for v in config.eval_names}

    model.eval()
    with torch.no_grad():
        for j, (_, image_identifiers, x, yp) in enumerate(dataloader):
            image_identifiers = np.array(image_identifiers).T
            batch_video_names = np.array([s.split('/')[0] for s in image_identifiers[:, 0]])
            x = x.cuda()
            yp = yp.cuda()

            g, _, _, _, _, _ = model(x)
            b = torch.sign(g.detach())

            for i in range(len(batch_video_names)):
                video_name = batch_video_names[i]
                bins_video[video_name].append(b[i].cpu())
                yps_video[video_name].append(yp[i].cpu())
                frame_ids_video[video_name].append(image_identifiers[i])

        for video_name in config.eval_names:
            bins_video[video_name] = torch.stack(bins_video[video_name], dim=0)
            yps_video[video_name] = torch.stack(yps_video[video_name], dim=0)
            frame_ids_video[video_name] = np.stack(frame_ids_video[video_name], axis=0)

        for video_name in config.eval_names:
            np.save('./data/hash_code' + f'/test_hash_{config.code_length}bits_{video_name}.npy', bins_video[video_name].numpy())
            np.save('./data/hash_code' + f'/test_label_{config.code_length}bits_{video_name}.npy', yps_video[video_name].numpy())
            np.save('./data/hash_code' + f'/test_path_{config.code_length}bits_{video_name}.npy', frame_ids_video[video_name])
            ex.add_artifact('./data/hash_code' + f'/test_hash_{config.code_length}bits_{video_name}.npy')
            ex.add_artifact('./data/hash_code' + f'/test_label_{config.code_length}bits_{video_name}.npy')
            ex.add_artifact('./data/hash_code' + f'/test_path_{config.code_length}bits_{video_name}.npy')


if __name__ == '__main__':
    ex.run(named_configs=[args.config])
