import torch
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import model as mdl
import time
from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,
)

from torch.utils.tensorboard.writer import SummaryWriter

device = "cpu"
torch.set_num_threads(4)

batch_size = 256  # batch for one node


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: CrossEntropyLoss,
    epoch: int,
    writer: SummaryWriter,
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

    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler("../logs/task_1/profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # remember to exit the train loop at end of the epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            # Your code goes here!
            # if batch_idx >= 40:
            #    break  # NOTE: Since shuffle is true we get different samples

            start_time = time.time()

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output: torch.Tensor = model(data)

            loss: torch.Tensor = criterion(output, target)

            loss.backward()

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
    train_loader: DataLoader = DataLoader(
        training_set,
        num_workers=2,
        batch_size=batch_size,
        sampler=None,
        shuffle=True,
        pin_memory=True,
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    test_loader: DataLoader = DataLoader(
        test_set, num_workers=2, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    training_criterion = CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(log_dir="../logs/task_1/rank_0")

    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch, writer)
        test_model(model, test_loader, training_criterion)


if __name__ == "__main__":
    main()
