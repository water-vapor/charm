from typing import Any, Optional
import hashlib
import os
import socket as socket_module

import coolname
import torch
import torch.distributed as dist

if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = "xpu"
    BACKEND = "xccl"
    from mpi4py import MPI
elif hasattr(torch, "cuda") and torch.cuda.is_available():
    DEVICE = "cuda"
    BACKEND = "nccl"
    MPI = None
else:
    raise ValueError("only cuda and xpu are supported")

SOCKET = socket_module


def init_distributed_environment(
    *,
    device: str = DEVICE,
    backend: str = BACKEND,
    mpi: Optional[Any] = MPI,
    socket: Any = SOCKET,
    log_xpu_master_addr: bool = False,
) -> tuple[int, int, int]:
    """
    Initialize torch.distributed for the active device.

    Returns:
        (rank, world_size, local_rank)
    """
    if device == "xpu":
        if mpi is None:
            raise ValueError("xpu initialization requires an MPI handle")

        # Import ipex after Hydra config loading to avoid OmegaConf parsing issues.
        import intel_extension_for_pytorch as ipex  # noqa: F401

        size = mpi.COMM_WORLD.Get_size()
        rank = mpi.COMM_WORLD.Get_rank()
        local_rank = int(os.environ.get("PALS_LOCAL_RANKID", "0"))
        world_size = size

        if size > 1:
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(size)
            os.environ["LOCAL_RANK"] = str(local_rank)

            master_addr = socket.gethostname() if rank == 0 else None
            if rank == 0 and log_xpu_master_addr:
                print(f"[Rank 0] socket.gethostname() = {master_addr}", flush=True)
            master_addr = mpi.COMM_WORLD.bcast(master_addr, root=0)
            if rank == 0 and log_xpu_master_addr:
                print(f"[Rank 0] After bcast: MASTER_ADDR = {master_addr}", flush=True)
            os.environ["MASTER_ADDR"] = f"{master_addr}.hsn.cm.aurora.alcf.anl.gov"

            # Use a job-specific port to avoid conflicts with other jobs.
            job_id = os.environ.get("PBS_JOBID", str(os.getpid()))
            port_hash = int(hashlib.md5(job_id.encode()).hexdigest()[:8], 16)
            master_port = 29500 + (port_hash % 1000)
            os.environ["MASTER_PORT"] = str(master_port)

            torch.xpu.set_device(local_rank)
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=rank,
                world_size=size,
            )
        else:
            torch.xpu.set_device(0)
        return rank, world_size, local_rank

    if device == "cuda":
        rank = 0
        world_size = 1
        local_rank = 0
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=backend, device_id=torch.device(f"cuda:{local_rank}"))
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            torch.cuda.set_device(0)
        return rank, world_size, local_rank

    raise ValueError("only cuda and xpu are supported")


def normalize_config_defaults(config: Any, run_name_suffix: str = "") -> None:
    """Apply shared config normalization used by pretrain and replay-eval."""
    if config.project_name is None:
        names = [os.path.splitext(os.path.basename(p))[0] for p in config.data_paths]
        config.project_name = "+".join(names)
    if config.run_name is None:
        config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
    if run_name_suffix and not config.run_name.endswith(run_name_suffix):
        config.run_name = f"{config.run_name}{run_name_suffix}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
    if config.eval_exp_decay:
        if config.eval_decay_half_life <= 0:
            raise ValueError(
                "eval_decay_half_life must be > 0 when eval_exp_decay=True, "
                f"got {config.eval_decay_half_life}"
            )


def broadcast_model_state(model: torch.nn.Module, world_size: int) -> None:
    if world_size <= 1:
        return
    with torch.no_grad():
        for param in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(param, src=0)
