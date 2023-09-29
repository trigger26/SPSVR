import torch

def save_checkpoint(path, epoch, model, optimizer, scheduler=None, centers=None, ex=None):
    save_dict = {'state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    if centers is not None:
        save_dict['center_state_dict'] = centers.state_dict()
    torch.save(save_dict, path)
    if ex is not None:
        ex.add_artifact(path)


def resume(path, model, optimizer, scheluder=None, centers=None, ex=None):
    point_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(point_dict['state_dict'])
    optimizer.load_state_dict(point_dict['optimizer_state_dict'])
    if scheluder is not None and 'scheduler_state_dict' in point_dict.keys():
        scheluder.load_state_dict(point_dict['scheduler_state_dict'])
    if centers is not None and 'center_state_dict' in point_dict.keys():
        centers.load_state_dict(point_dict['center_state_dict'])
    if ex is not None:
        ex.add_resource(path)
    print(f"Load checkpoint from {path}, epoch: {point_dict['epoch']}")
    return point_dict['epoch']
