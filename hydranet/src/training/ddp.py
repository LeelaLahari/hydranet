"""Single-node DistributedDataParallel hook (SLURM/HPC-ready).

Colab free tier = 1 GPU, so this is dormant by default; document it in the
README/report to show HPC awareness. Activate with:
    torchrun --nproc_per_node=NGPU scripts/03_train_lstm.py --ddp
"""
import os, torch, torch.distributed as dist

def init_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True, int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    return False, 1, 0

def wrap_ddp(model, is_ddp):
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[torch.cuda.current_device()]) if is_ddp else model

def cleanup_ddp(is_ddp):
    if is_ddp:
        dist.destroy_process_group()
