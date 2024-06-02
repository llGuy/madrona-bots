import os
import torch

class CheckpointManager:
    def __init__(self, base_ckpt_dir, restore=True):
        self.base_ckpt_dir = base_ckpt_dir
        self.restore = restore
        self._check_dir(base_ckpt_dir)

    def save(self, model, optimizer, sub_dir, epoch):
        full_path = os.path.join(self.base_ckpt_dir, sub_dir)
        self._check_dir(full_path)
        
        save_path = os.path.join(full_path, f'model_epoch_{epoch}.pt')
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': model.get_config()
        }, save_path)

    def load(self, model_class, optimizer_class, sub_dir, epoch, init_fn):
        load_path = os.path.join(self.base_ckpt_dir, sub_dir, f'model_epoch_{epoch}.pt')
        assert init_fn is not None, "Init function needed"

        if self.restore and os.path.isfile(load_path):
            checkpoint = torch.load(load_path)
            model = init_fn(checkpoint['model_config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = optimizer_class(model.parameters())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return model, optimizer
        else:
            model = init_fn()
            optimizer = optimizer_class(model.parameters(), lr=3e-4)
            return model, optimizer

    def _check_dir(self, path):
        # Create the directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)