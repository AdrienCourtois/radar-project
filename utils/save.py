import os
import torch
import shutil

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    """
    Saves entire model.

    state dictionnary template:
    {'epoch': epoch,
    'state_dict': model.state_dict(),
    'best_IOU': best_IOU,
    'optimizer': optimizer.state_dict(),
    'iter': cpt}

    """
    path = os.path.join(save_path, filename)
    torch.save(state, path)
    if is_best:
        path2 = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(path, path2)
