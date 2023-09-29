import torch

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricLogger:
    def __init__(self, meters = {}):
        self.meters = meters

    def update(self, metric_dict=None, **kwargs):
        if metric_dict is not None:
            self._update_item(metric_dict, self.meters)
        else:
            self._update_item(kwargs, self.meters)

    def _update_item(self, metric_dict, log_dict):
        for k, v in metric_dict.items():
            if k not in self.meters.keys():
                log_dict[k] = AverageMeter()
            
            if isinstance(v, torch.Tensor):
                log_dict[k].update(v.item())
            elif isinstance(v, float) or isinstance(v, int):
                log_dict[k].update(v)
            elif isinstance(v, dict):
                log_dict[k] = {}
                self._update_item(v, log_dict[k])
            else:
                raise ValueError

    def get_metrics(self):
        return self._get_items(self.meters)

    def _get_items(self, meters_dict):
        res = {}
        for k, v in meters_dict.items():
            if isinstance(v, dict):
                res[k] = self._get_items(v)
            else:
                res[k] = v.avg
        return res

    def get(self, key):
        keys = key.split("/")
        res = self.meters
        for k in keys:
            res = res[k]
        return res.avg

    def __getattr__(self, key):
        res = self.meters[key]
        if isinstance(res, dict):
            return MetricLogger(res)
        return res.avg


def log_metrics(metrics: dict, epoch, sacred_ex=None, tb_writer=None):
    parsed = {}

    def r_parse(metrics: dict, key_pref=''):
        for k, v in metrics.items():
            full_key = '/'.join([key_pref, k]) if key_pref != '' else k
            if isinstance(v, dict):
                r_parse(v, full_key)
            else:
                parsed[full_key] = v

    r_parse(metrics)
    for k, v in parsed.items():
        if sacred_ex is not None:
            sacred_ex.log_scalar(k, v, epoch)
        if tb_writer is not None:
            tb_writer.add_scalar(k, v, epoch)