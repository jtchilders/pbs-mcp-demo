#!/usr/bin/env python3
"""Distributed PyTorch training example for Aurora (Intel GPUs).

This script demonstrates distributed data-parallel training on Aurora's
Intel Data Center GPU Max series using PyTorch with Intel Extension for PyTorch (IPEX).

Features:
- Multi-node, multi-GPU distributed training
- Intel XPU (GPU) support via intel_extension_for_pytorch
- oneCCL backend for efficient collective communications
- Automatic mixed precision (AMP) for performance
- Simple ResNet-style model training on synthetic data

Usage:
    This script is designed to be launched via mpiexec on Aurora:

    mpiexec -n 24 -ppn 12 python train_resnet_aurora.py --epochs 10

    Or use the generate_aurora_pytorch_script tool to create a PBS submit script.
"""

import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False
    print("Warning: intel_extension_for_pytorch not available")

# oneCCL for distributed communication
try:
    import oneccl_bindings_for_pytorch  # noqa: F401
    HAS_CCL = True
except ImportError:
    HAS_CCL = False


class SyntheticDataset(Dataset):
    """Synthetic dataset for benchmarking distributed training."""

    def __init__(self, num_samples: int = 10000, image_size: int = 224, num_classes: int = 1000):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Generate random image and label
        image = torch.randn(3, self.image_size, self.image_size)
        label = idx % self.num_classes
        return image, label


class SimpleResNet(nn.Module):
    """A simplified ResNet-like model for demonstration."""

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Simplified residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training environment.

    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    # Get MPI rank information from environment
    rank = int(os.environ.get("PMI_RANK", os.environ.get("RANK", 0)))
    local_rank = int(os.environ.get("PALS_LOCAL_RANKID", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("PMI_SIZE", os.environ.get("WORLD_SIZE", 1)))

    # Initialize process group
    if world_size > 1:
        backend = "ccl" if HAS_CCL else "gloo"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=world_size,
                rank=rank
            )
        if rank == 0:
            print(f"Distributed training initialized: {world_size} processes, backend={backend}")

    return rank, local_rank, world_size


def get_device(local_rank: int) -> torch.device:
    """Get the appropriate device for this rank."""
    if torch.xpu.is_available():
        device = torch.device(f"xpu:{local_rank % torch.xpu.device_count()}")
        torch.xpu.set_device(device)
        return device
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        return device
    else:
        return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    rank: int,
    use_amp: bool = True,
) -> float:
    """Train for one epoch.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward pass (with optional AMP)
        if use_amp and device.type == "xpu":
            with torch.xpu.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
        else:
            output = model(data)
            loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log progress
        if rank == 0 and batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    avg_loss = total_loss / num_batches

    if rank == 0:
        throughput = len(loader.dataset) / elapsed
        print(f"Epoch {epoch} completed in {elapsed:.2f}s, "
              f"Avg Loss: {avg_loss:.4f}, Throughput: {throughput:.1f} samples/sec")

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Distributed PyTorch Training on Aurora")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num-samples", type=int, default=10000, help="Synthetic dataset size")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    args = parser.parse_args()

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    device = get_device(local_rank)

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Aurora Distributed PyTorch Training")
        print(f"{'=' * 60}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"IPEX available: {HAS_IPEX}")
        print(f"CCL available: {HAS_CCL}")
        print(f"XPU available: {torch.xpu.is_available() if hasattr(torch, 'xpu') else False}")
        print(f"Batch size per device: {args.batch_size}")
        print(f"Global batch size: {args.batch_size * world_size}")
        print(f"{'=' * 60}\n")

    # Create model
    model = SimpleResNet(num_classes=args.num_classes).to(device)

    # Loss and optimizer (must be created before ipex.optimize)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # Optimize with IPEX if available (must pass both model and optimizer for training)
    if HAS_IPEX and device.type == "xpu":
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

    # Wrap model for distributed training
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Create synthetic dataset and dataloader
    dataset = SyntheticDataset(
        num_samples=args.num_samples,
        num_classes=args.num_classes
    )

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,  # Avoid AF_UNIX path issues on Aurora
        pin_memory=False,
    )

    # Training loop
    use_amp = not args.no_amp
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_epoch(model, loader, criterion, optimizer, device, epoch, rank, use_amp)

    total_time = time.time() - total_start

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Training completed in {total_time:.2f}s")
        print(f"Average time per epoch: {total_time / args.epochs:.2f}s")
        print(f"{'=' * 60}")

    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
