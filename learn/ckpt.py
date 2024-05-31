import os

class CheckpointManager:
    def __init__(self, ckpt_dir, restore):
        self.ckpt_dir = ckpt_dir
        self.restore = restore

        _check_ckpt_dir(ckpt_dir)

    # Gets the policy networks for the species from the checkpoint directory.
    # If the directory doesn't contain anything, or restore is set to false,
    # we need to initialize random weights
    def load_networks(self, init_fn)
        if self.restore and len(os.listdir(self.ckpt_dir)) > 0:
            raise NotImplementedError
        else:
            return init_fn()

    def save(self):
        pass

    def _check_ckpt_dir(self, ckpt_dir):
        # This will create the directory if it doesn't exit
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
