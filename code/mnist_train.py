import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm

from utils import setup_logging, adjust_precision, PminContext


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_epoch(
    args,
    model,
    device: str,
    train_loader,
    optimizer,
    *,
    pmin_ctx: PminContext,
    dynomp: bool = False,
    amp: bool = False,
):
    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        current_dtype = next(model.parameters()).dtype
        # Cast inputs
        data = data.to(device, dtype=current_dtype)
        # Target must remain as initial dtype (i.e., long) for loss computation
        target = target.to(device)

        optimizer.zero_grad()

        if amp:
            with torch.autocast(device_type=device):
                output = model(data)
                loss = F.nll_loss(output, target)
        else:
            output = model(data)
            loss = F.nll_loss(output, target)

        loss.backward()

        if dynomp:
            # Compute the required precision bits
            adjust_precision(
                model,
                pmin_ctx=pmin_ctx,
                ith_step=batch_idx,
                optimizer=optimizer,
            )

        # Resume training with the selected precision
        optimizer.step()
        total_loss += loss.item()

        # Log training progress
        logging.debug(
            ", ".join(
                [
                    f"Step {batch_idx + 1}",
                    f"Step loss = {loss.item():.4f}",
                ]
            )
        )

    return total_loss / len(train_loader.dataset)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            current_dtype = next(model.parameters()).dtype
            data = data.to(device, dtype=current_dtype)

            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logging.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input data directory"
    )
    parser.add_argument(
        "--output", type=str, default="model.pt", help="Output model path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="default",
        help="Training strategy to use (default, dynomp, amp, fp16, or bf16)",
    )
    args = parser.parse_args()

    # Setup logging
    logs = Path("logs", "mnist")
    logs.mkdir(exist_ok=True, parents=True)
    log_file = Path(logs / f"mnist_train_{args.strategy}.log")
    setup_logging(log_file)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Create model
    model = Net().to(device)

    # BF16 training strategy: Set model and optimizer to BF16 precision
    if args.strategy == "bf16":
        model = model.to(torch.bfloat16)
        logging.info("Using BF16 precision for training")
    # FP16 training strategy: Set model and optimizer to FP16 precision
    if args.strategy == "fp16":
        model = model.to(torch.float16)
        logging.info("Using FP16 precision for training")

    # PMIN context
    pmin_ctx = PminContext()

    # Setup losses and optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Create dataloaders
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST(args.input, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(args.input, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Training loop
    logging.info(f"Training for {args.epochs} epochs...")
    test(model, device, test_loader)
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        avg_loss = train_epoch(
            args,
            model,
            device,
            train_loader,
            optimizer,
            pmin_ctx=pmin_ctx,
            dynomp=(args.strategy == "dynomp"),
            amp=(args.strategy == "amp"),
        )
        # Log progress
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}")
        test(model, device, test_loader)
        scheduler.step()

    # Save final model
    torch.save(model.state_dict(), output_path)
    logging.info(f"Final model saved to {output_path}")


if __name__ == "__main__":
    main()
