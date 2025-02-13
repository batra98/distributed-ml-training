from typing import cast
import torch
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import model as mdl
import time
import argparse
import torch.distributed as dist
import numpy as np


device = "cpu"
torch.set_num_threads(4)

batch_size = 64  # batch for one node
seed = 42


def init_distributed_mode(args):
    if args.rank == 0:
        print("Initializing distributed mode...")

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{args.master_ip}:29500",
        world_size=args.num_nodes,
        rank=args.rank,
    )

    torch.manual_seed(seed)


def sync_gradients_allreduce(model):
    """Synchronize gradients using AllReduce."""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)  # Sum all gradients
            param.grad /= dist.get_world_size()  # Average gradients


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: CrossEntropyLoss,
    epoch: int,
    writer: SummaryWriter | None,
):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()  ### just a good practice, only necessary if we have Droupout/BatchNorm layers
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    times = []

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "../logs_40/task_2b/profiler"
        ),
        record_shapes=True,
        with_stack=True,
    ) as prof:

        # remember to exit the train loop at end of the epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            # Your code goes here!
            if batch_idx >= 40:
                break  # NOTE: Since shuffle is true we get different samples

            start_time = time.time()

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output: torch.Tensor = model(data)
            loss: torch.Tensor = criterion(output, target)

            loss.backward()
            sync_gradients_allreduce(model)
            optimizer.step()

            end_time = time.time()
            iteration_time = end_time - start_time

            times.append(iteration_time)

            running_loss += loss.item()

            _, predicted = output.max(1)
            correct += int(predicted.eq(target).sum().item())
            total += target.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch+1}], Iteration [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

            if dist.get_rank() == 0:
                writer = cast(SummaryWriter, writer)
                writer.add_scalar(
                    "Train/Loss", loss.item(), epoch * len(train_loader) + batch_idx
                )
                writer.add_scalar(
                    "Train/Iteration Time",
                    iteration_time,
                    epoch * len(train_loader) + batch_idx,
                )
                for name, param in model.named_parameters():
                    writer.add_histogram(f"Params/{name}", param, epoch)
                    writer.add_histogram(f"Grads/{name}", param.grad, epoch)

            prof.step()

    avg_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else 0.0
    print(
        f"Epoch [{epoch+1}] completed. Average time per iteration (after discarding first): {avg_time:.6f} seconds"
    )

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(
        f"Epoch [{epoch+1}] Training complete. Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
    )

    if dist.get_rank() == 0:
        writer = cast(SummaryWriter, writer)
        writer.add_scalar("Train/Avg Loss", avg_loss, epoch)
        writer.add_scalar("Train/Accuracy", accuracy, epoch)
        writer.add_scalar("Train/Avg Iteration Time", avg_time, epoch)


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: CrossEntropyLoss,
):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master-ip", required=True, type=str, help="IP address of the master node"
    )
    parser.add_argument(
        "--num-nodes",
        required=True,
        type=int,
        help="Total number of nodes in the distributed setup",
    )
    parser.add_argument(
        "--rank",
        required=True,
        type=int,
        help="Rank of the current node (starting from 0)",
    )
    args = parser.parse_args()

    init_distributed_mode(args)

    torch.manual_seed(seed)
    np.random.seed(seed)

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    training_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_sampler = DistributedSampler(
        training_set, num_replicas=args.num_nodes, rank=args.rank
    )
    train_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    model = mdl.VGG11().to(device)

    criterion = CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    writer = (
        SummaryWriter(log_dir=f"../logs_40/task_2b/rank_{args.rank}")
        if args.rank == 0
        else None
    )

    # Training loop
    for epoch in range(1):
        train_sampler = cast(
            DistributedSampler, train_sampler
        )  # Just to remove typing error
        train_sampler.set_epoch(epoch)
        train_model(model, train_loader, optimizer, criterion, epoch, writer)
        test_model(model, test_loader, criterion)


if __name__ == "__main__":
    main()
