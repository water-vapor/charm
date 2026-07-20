import os
import torch.distributed as dist


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)
