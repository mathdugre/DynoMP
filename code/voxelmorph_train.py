#!/usr/bin/env python3
"""
You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8).
    pp 1788-1800. 2019.

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. MedIA: Medical Image Analysis. (57).
    pp 226-236, 2019

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

# Core library imports
import argparse
from typing import Sequence
from pathlib import Path
import logging

# Third-party imports
import numpy as np
import nibabel as nib
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import neurite as ne

# Local imports
import voxelmorph as vxm
from utils import adjust_precision, PminContext, setup_logging


class OASIS3D(IterableDataset):
    """
    PyTorch IterableDataset for infinite VoxelMorph registration data.
    """

    def __init__(self, path: Path, device: str = "cpu") -> None:
        """
        Parameters
        ----------
        device : str
            Device to place tensors on.
        """
        self.device = device
        self.oasis_path = path
        self.norm_file = "aligned_norm.nii.gz"
        self._get_vol_paths()

    def __iter__(self):
        """
        Generate infinite stream of random volume pairs.

        Yields
        ------
        dict
            A dictionary containing the source and target volumes.
        """
        while True:
            idx1, idx2 = np.random.randint(0, len(self.folder_abspaths), size=2)

            # Get paths
            source_path = self.folder_abspaths[idx1]
            target_path = self.folder_abspaths[idx2]

            # Get niftis
            source_nii = nib.load(f"{source_path}/{self.norm_file}")
            target_nii = nib.load(f"{target_path}/{self.norm_file}")

            # Transform to

            # Get data and convert to tensors
            source = torch.from_numpy(source_nii.get_fdata()).float().unsqueeze(0)
            target = torch.from_numpy(target_nii.get_fdata()).float().unsqueeze(0)

            yield {"source": source, "target": target}

    def _get_vol_paths(self) -> None:
        """
        Get the absolute paths of the volume folders.
        """
        self.folder_abspaths = []

        for i in range(1, 450):
            folder = self.oasis_path / f"OASIS_OAS1_{i:04}_MR1"

            if folder.exists():
                self.folder_abspaths.append(folder)


class OASIS2D(OASIS3D):
    """
    PyTorch IterableDataset for infinite VoxelMorph registration data.
    """

    def __init__(self, path: Path, device: str = "cpu") -> None:
        """
        Parameters
        ----------
        device : str
            Device to place tensors on.
        """
        self.device = device
        self.oasis_path = path
        self.norm_file = "slice_norm.nii.gz"
        self._get_vol_paths()

    def __iter__(self):
        """
        Generate infinite stream of random volume pairs.

        Yields
        ------
        dict
            A dictionary containing the source and target volumes.
        """
        while True:
            idx1, idx2 = np.random.randint(0, len(self.folder_abspaths), size=2)

            # Get paths
            source_path = self.folder_abspaths[idx1]
            target_path = self.folder_abspaths[idx2]

            # Get niftis
            source_nii = nib.load(f"{source_path}/{self.norm_file}")
            target_nii = nib.load(f"{target_path}/{self.norm_file}")

            # Transform to

            # Get data and convert to tensors
            source = (
                torch.from_numpy(source_nii.get_fdata()[:, :, 0]).float().unsqueeze(0)
            )
            target = (
                torch.from_numpy(target_nii.get_fdata()[:, :, 0]).float().unsqueeze(0)
            )

            yield {"source": source, "target": target}


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    steps_per_epoch: int,
    pmin_ctx: PminContext,
    device: str = "cuda",
    dynomp: bool = False,
    amp: bool = False,
) -> float:
    """
    Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The VoxelMorph model to train.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use for training.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    image_loss_fn : nn.Module
        The image loss function to use.
    grad_loss_fn : nn.Module
        The gradient loss function to use.
    loss_weights : Sequence[float]
        The weights for the image and gradient losses.
    steps_per_epoch : int
        The number of steps to train for in one epoch.
    pmin_ctx : PminContext
        The PMIN context for dynamic mixed precision.
    device : str, optional
        The device to train on, by default "cuda".
    dynomp : bool, optional
        Whether to use Dynamic Mixed Precision (DynoMP) variant of VoxelMorph,
        by default False.
    amp: bool, optional
        Whether to use Automatic Mixed Precision (AMP) for training, by default False.
    """
    model.train()
    total_loss = 0.0

    for ith_step in range(steps_per_epoch):
        batch = next(dataloader)

        current_dtype = next(model.parameters()).dtype
        # Cast inputs
        source = batch["source"].to(device, dtype=current_dtype)
        target = batch["target"].to(device, dtype=current_dtype)

        optimizer.zero_grad()

        # Get the displacement and the warped source image from the model
        if amp:
            with torch.autocast(device_type=device):
                displacement, warped_source = model(
                    source,
                    target,
                    return_warped_source=True,
                    return_field_type="displacement",
                )
                img_loss = image_loss_fn(target, warped_source)
                grad_loss = grad_loss_fn(displacement)
        else:
            displacement, warped_source = model(
                source,
                target,
                return_warped_source=True,
                return_field_type="displacement",
            )
            img_loss = image_loss_fn(target, warped_source)
            grad_loss = grad_loss_fn(displacement)
            
        loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
        loss.backward()

        # DynoMP Strategy: Adjust precision based on PMIN moving average
        if dynomp:
            dtype = adjust_precision(
                model,
                pmin_ctx=pmin_ctx,
                ith_step=ith_step,
                optimizer=optimizer,
            )
            if current_dtype != dtype:
                # Force reallocate meshgrid when dtype changes
                del model.spatial_transformer.meshgrid

        optimizer.step()
        total_loss += loss.item()

        # Log training progress
        logging.debug(
            ", ".join(
                [
                    f"Step {ith_step + 1}",
                    f"Step loss = {loss.item():.4f}",
                    f"Image loss = {img_loss.item():.4f}",
                    f"Grad loss = {grad_loss.item():.4f}",
                ]
            )
        )

    return total_loss / steps_per_epoch


def main():
    parser = argparse.ArgumentParser(description="Train VoxelMorph on OASIS data")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input data directory"
    )
    parser.add_argument(
        "--output", type=str, default="model.pt", help="Output model path"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers")
    parser.add_argument(
        "--steps-per-epoch", type=int, default=100, help="Steps per epoch"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda", type=float, dest="lambda_param", default=0.01)
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument(
        "--save-every", type=int, default=10, help="Checkpoint every N epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0, help="Early stopping threshold"
    )
    parser.add_argument(
        "--warm-start", type=int, default=10, help="Early stopping warm start steps"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dim", type=int, default=3, help="Dimension of the data (2 or 3)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="default",
        help="Training strategy to use (default, dynomp, amp, fp16, or bf16)",
    )
    args = parser.parse_args()

    # Setup logging
    logs = Path("logs", "voxelmorph")
    logs.mkdir(exist_ok=True, parents=True)
    log_file = Path(logs / f"voxelmorph_train_{args.dim}d_{args.strategy}.log")
    setup_logging(log_file)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Create model
    model = vxm.nn.models.VxmPairwise(
        ndim=args.dim,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 16, 16, 16, 16],
        integration_steps=0,
    ).to(device)

    # BF16 training strategy: Set model and optimizer to BF16 precision
    if args.strategy == "bf16":
        model = model.to(torch.bfloat16)
        logging.info("Using BF16 precision for training")
    if args.strategy == "fp16":
        model = model.to(torch.float16)
        logging.info("Using FP16 precision for training")

    # PMIN context
    pmin_ctx = PminContext()

    # Setup losses and optimizer
    image_loss_fn = ne.nn.modules.MSE()
    grad_loss_fn = ne.nn.modules.SpatialGradient("l2")
    loss_weights = [1.0, args.lambda_param]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create dataloader
    input_dataset = Path(args.input)
    if args.dim == 3:
        train_dataset = OASIS3D(input_dataset, device=device)
    elif args.dim == 2:
        train_dataset = OASIS2D(input_dataset, device=device)
    else:
        raise ValueError("Dimension must be 2 or 3")
    train_loader = iter(
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    )

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Training loop
    logging.info(f"Training for {args.epochs} epochs...")
    best_loss = float("inf")
    loss_history = []
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # Train for one epoch
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            steps_per_epoch=args.steps_per_epoch,
            pmin_ctx=pmin_ctx,
            device=device,
            dynomp=(args.strategy == "dynomp"),
            amp=(args.strategy == "amp"),
        )

        # Track loss history
        loss_history.append(avg_loss)

        # Log progress
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}")

        # Early stopping check
        if ne.utils.early_stopping(
            loss_history,
            patience=args.patience,
            threshold=args.threshold,
            warm_start_steps=args.warm_start,
        ):
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:
            # Build checkpoint file name
            checkpoint_path = output_path.parent.joinpath(
                f"{output_path.stem}_default-int_epoch{epoch + 1}.pt"
            )

            # Save
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_path.parent / f"{output_path.stem}_best.pt"
            torch.save(model.state_dict(), best_path)

    # Save final model
    torch.save(model.state_dict(), args.output)
    logging.info(f"Final model saved to {args.output}")


if __name__ == "__main__":
    main()
