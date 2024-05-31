import argparse
import dataclasses

from util import set_seed
from ckpt import CheckpointManager

@dataclass
class RunConfig:
    gpu_id: int
    ckpt_dir: str
    restore: bool
    num_worlds: int
    lr: float
    gamma: float
    seed: int

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--gpu-id', type=int, default=0)
    arg_parser.add_argument('--ckpt-dir', type=str, required=True)
    arg_parser.add_argument('--restore', type=int, required=True)

    arg_parser.add_argument('--num-worlds', type=int, required=True)

    arg_parser.add_argument('--lr', type=float, default=1e-4)
    arg_parser.add_argument('--gamma', type=float, default=0.998)
    arg_parser.add_argument('--seed', type=int, default=0)

    args = arg_parser.parse_args()

    return RunConfig(
        gpu_id=args.gpu_id,
        ckpt_dir=args.ckpt_dir,
        restore=args.restore,
        num_worlds=args.num_worlds,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed
    )

def main():
    run_cfg = parse_args()
    set_seed(run_cfg.seed)

    ckpt_mgr = CheckpointManager(run_cfg.ckpt_dir, run_cfg.restore)

if __name__ == "__main__":
    main()
