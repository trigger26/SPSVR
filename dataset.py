import os
import csv
import numpy as np
from operator import itemgetter

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.io import read_image


_FPS_CHOLEC80 = 25
_FPS_CATARACTS = 25


class VideoClip(Dataset):
    def __init__(self, img_root, ann_path, num_class, video_name, n_frame, duration, fps, group_tfs=None, drop_phase_transition=False):
        self.img_root = img_root
        self.video_name = video_name
        self.n_frame = n_frame
        self.group_tfs = group_tfs

        frames = self.load_csv(ann_path, video_name)
        clips = self.clip_devide(frames, duration, fps)
        if drop_phase_transition:
            clips = self.filter_wo_phase_transition(clips)
        self.postprocess_label(clips, num_class)
        self.clips = clips


    def __getitem__(self, index):
        clip = self.clips[index]
        phase = clip["phase"]

        img_paths = clip["img"]        
        img_ids = clip["img_id"]

        # downsample
        inds = self.uniform_subsample(self.n_frame, len(img_paths))
        img_paths = img_paths[inds]
        img_ids = img_ids[inds]

        # read & transform
        imgs = self.group_tfs(
            [read_image(os.path.join(self.img_root, path)) for path in img_paths])

        return index, list(img_ids), imgs, phase


    def __len__(self):
        return len(self.clips)
    

    def load_csv(self, ann_path, video_name):
        res = []
        csv_path = os.path.join(ann_path, video_name + '.csv')
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                res.append({
                    "img": row[0],
                    "phase": int(row[1]),
                    "img_id": row[0].split('.')[0]
                })
        return res
    

    def uniform_subsample(self, sample_n, length):
        return np.linspace(0, length-1, sample_n, dtype=int)


    def filter_wo_phase_transition(self, clips):
        res = []
        for clip in clips:
            if clip["phase"].min() == clip["phase"].max():
                res.append(clip.copy())
        return res


    def postprocess_label(self, clips, num_classes):
        """ array[x, x, x] -> one hot """
        for clip in clips:
            label = int(clip["phase"][0])
            label = one_hot(label, num_classes)  # one hot
            clip["phase"] = label


    def clip_devide(self, frames, duration, fps):
        """ video to clips

            Args:
                frames: video frames metadata List
                duration: clip duration (seconds)
            
            Return: Clip metadata List
        """
        res = []
        start = 0
        while start <= len(frames) - duration * fps:
            end = start + duration * fps
            clip = frames[start: end]
            keys = clip[0].keys()  # ["img", "phase", "img_id"]
            clip_numpy = {}
            for k in keys:
                clip_numpy[k] = np.array(list(map(itemgetter(k), clip)))
            res.append(clip_numpy)
            start += duration * fps
        return res


def one_hot(a, num_classes):
    if isinstance(a, int):
        b = np.zeros(num_classes)
        b[a] = 1
        return b
    elif isinstance(a, np.ndarray):
        b = np.zeros((a.shape[0], num_classes))
        b[np.arange(a.shape[0]), a.astype(int)] = 1
        return b
    else:
        raise ValueError()


def _get_multi_video_dataset_cholec80(img_root, ann_path, num_class, video_names, n_frame, duration, group_tfs, drop_phase_transition):
    datasets = []
    for video_name in video_names:
        datasets.append(VideoClip(img_root, ann_path, num_class, video_name, n_frame, duration, _FPS_CHOLEC80, group_tfs, drop_phase_transition))
    return ConcatDataset(datasets)


def _get_multi_video_dataset_cataract101(img_root, ann_path, num_class, video_names, n_frame, duration, group_tfs, drop_phase_transition):
    datasets = []
    for video_name in video_names:
        datasets.append(VideoClip(img_root, ann_path, num_class, video_name, n_frame, duration, _FPS_CATARACTS, group_tfs, drop_phase_transition))
    return ConcatDataset(datasets)


def get_dataloader(name, batch_size, shuffle, pin_memory, num_workers, **kwargs):
    assert name in ['cholec80', 'cataract101']

    if name == 'cholec80':
        dataset = _get_multi_video_dataset_cholec80(
            img_root=kwargs['data_dir'],
            ann_path=kwargs['label_dir'],
            num_class=kwargs['num_class'],
            video_names=kwargs['video_names'],
            n_frame=kwargs['n_frame'],
            duration=kwargs['duration'],
            group_tfs=kwargs['group_tfs'],
            drop_phase_transition=True,
        )
    elif name == 'cataract101':
        dataset = _get_multi_video_dataset_cataract101(
            img_root=kwargs['data_dir'],
            ann_path=kwargs['label_dir'],
            num_class=kwargs['num_class'],
            video_names=kwargs['video_names'],
            n_frame=kwargs['n_frame'],
            duration=kwargs['duration'],
            group_tfs=kwargs['group_tfs'],
            drop_phase_transition=True
        )

    return DataLoader(dataset, batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
