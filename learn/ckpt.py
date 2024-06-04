import os
import torch
import glob
import fnmatch


class CheckpointManager:
    def __init__(self, base_ckpt_dir, restore=True):
        self.base_ckpt_dir = base_ckpt_dir
        self.restore = restore
        self._check_dir(base_ckpt_dir)

    def save(self, model, optimizer, sub_dir, epoch, metric_name="latest"):
        full_path = os.path.join(self.base_ckpt_dir, sub_dir)
        self._check_dir(full_path)
        
        # Determine the filename based on whether it's the latest or a best metric model
        if metric_name == "latest":
            filename = f'latest_model_epoch_{epoch}.pt'
        else:
            filename = f'best_{metric_name}_epoch_{epoch}.pt'
            # Delete any existing best model files for the same metric
            self._delete_old_files(full_path, f'best_{metric_name}_epoch_*.pt')

        save_path = os.path.join(full_path, filename)
        
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model.get_config()
            }, save_path)
            print(f"Saved model to {save_path}")
        except Exception as e:
            print(f"Failed to save model due to error: {e}")

    def load(self, model_class, optimizer_class, sub_dir, init_fn, metric_name="latest", epoch=None):
        if metric_name == "latest":
            assert epoch is not None, "Epoch must be specified for loading the latest model"
            load_path = os.path.join(self.base_ckpt_dir, sub_dir, f'latest_model_epoch_{epoch}.pt')
        else:
            # Load the best model for the specified metric
            load_path = os.path.join(self.base_ckpt_dir, sub_dir, f'best_{metric_name}_epoch_*.pt')
            files = glob.glob(load_path)
            if not files:
                raise FileNotFoundError(f"No best model found for metric {metric_name}")
            # Assuming the latest best model is needed
            files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
            load_path = files[0]
        
        if self.restore and os.path.isfile(load_path):
            checkpoint = torch.load(load_path)
            model = init_fn(checkpoint['model_config']) if init_fn else model_class(checkpoint['model_config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = optimizer_class(model.parameters())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return model, optimizer
        else:
            model = init_fn() if init_fn else model_class()
            optimizer = optimizer_class(model.parameters(), lr=3e-4)
            return model, optimizer

    def _check_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _delete_old_files(self, directory, pattern):
        for f in os.listdir(directory):
            if fnmatch.fnmatch(f, pattern):
                os.remove(os.path.join(directory, f))
