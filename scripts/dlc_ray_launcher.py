"""DLC Ray cluster launcher for multi-node veRL training on Alibaba Cloud.

DLC sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT on each container.
RANK=0 starts Ray head; others start Ray workers and sleep forever.
Only RANK=0 proceeds to run the training command.

Usage:
    python scripts/dlc_ray_launcher.py -- python scripts/train_multi_episode.py ...
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import ray


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    # Separate port for Ray, avoid collision with torch.distributed MASTER_PORT
    ray_port = os.environ.get("RAY_PORT", "16379")

    print(f"[DLC] rank={rank}, world_size={world_size}, "
          f"master_addr={master_addr}, ray_port={ray_port}")

    # ── Start Ray node ──────────────────────────────────────────────────────
    if rank == 0:
        print(f"[DLC] Starting Ray head on {master_addr}:{ray_port}")
        subprocess.run(
            f"ray start --head --port={ray_port} --node-ip-address={master_addr}",
            shell=True, check=True,
        )
    else:
        head_addr = f"{master_addr}:{ray_port}"
        for i in range(30):
            print(f"[DLC] Worker {rank} connecting to {head_addr} (attempt {i + 1})")
            ret = subprocess.run(f"ray start --address={head_addr}", shell=True)
            if ret.returncode == 0:
                break
            time.sleep(10)
        else:
            print("[DLC] ERROR: Worker failed to connect to head!")
            sys.exit(1)

    # ── Head waits for all nodes ────────────────────────────────────────────
    if rank == 0:
        print(f"[DLC] Waiting for {world_size} nodes...")
        ray.init(address="auto")
        for attempt in range(1, 121):
            alive = [n for n in ray.nodes() if n["Alive"]]
            if len(alive) >= world_size:
                print(f"[DLC] Ray cluster ready: {len(alive)}/{world_size} nodes.")
                break
            print(f"[DLC] Attempt {attempt}: {len(alive)}/{world_size} nodes...")
            time.sleep(10)
        else:
            print("[DLC] ERROR: Timed out waiting for Ray cluster!")
            sys.exit(1)
        ray.shutdown()

    # ── Workers sleep forever (DLC kills all containers on task end) ────────
    if rank != 0:
        print(f"[DLC] Worker {rank} standing by.")
        while True:
            time.sleep(3600)

    # ── Head: run training command (everything after "--") ──────────────────
    if "--" not in sys.argv:
        print("[DLC] ERROR: No command. Usage: python dlc_ray_launcher.py -- <cmd>")
        sys.exit(1)

    cmd = sys.argv[sys.argv.index("--") + 1:]
    print(f"[DLC] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    subprocess.run("ray stop", shell=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
