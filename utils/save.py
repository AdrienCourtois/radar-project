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
        
def load_checkpoint(model, optimizer, resume, mismatch=False):
    start_epoch = 0
    best_IOU = 0
    cpt = 1
    if resume:
        if os.path.isfile(resume):
            print(f"Loading checkpoint: {resume}.")
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_IOU = checkpoint['best_IOU']
            if mismatch:
              del checkpoint['state_dict']['DownConv0.conv.0.weight']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            cpt = checkpoint['iter']
            print(f"Checkpoint loaded: Start epoch: {start_epoch}, Best IOU: {best_IOU:0.3f}.")
        else:
            print(f"No checkpoint found at: {resume}.")
    else:
        print(f"No checkpoint loaded.")
    return model, optimizer, start_epoch, best_IOU, cpt 
