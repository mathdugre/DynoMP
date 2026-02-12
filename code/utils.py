from dataclasses import dataclass
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.linalg

EPSILON = 1e-9
WINDOW_SIZE = 5


@dataclass
class PminContext:
    prev_grad = None
    prev_param = None
    pmin_window = np.empty(WINDOW_SIZE)
    pmin_window.fill(np.nan)


def reset_logging():
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()


def setup_logging(log_file: Path, *, reset: bool = False):
    if reset:
        reset_logging()
    log_file.unlink(missing_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="{asctime} - {levelname} - {message}",
        style="{",  # This tells Python to look for {} instead of %()
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def _sync_optimizer(optimizer, model, dtype):
    # Sync model gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = p.grad.data.to(dtype=dtype)

    # Sync optimizer internal states
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and k != "step":
                state[k] = v.to(dtype=dtype)


def adjust_precision(
    model,
    /,
    *,
    pmin_ctx,
    ith_step,
    optimizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adjust the precision of the model and data based on the PMIN moving average.

    Args:
        model: The PyTorch model whose precision is to be adjusted.
        pmin_ctx: A dictionary containing previous gradients, parameters, and PMIN window.
        ith_step: The current training step index.
        optimizer: The optimizer used for training.
    """
    curr_grads = torch.cat(
        [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
    )
    curr_params = torch.cat([p.detach().view(-1) for p in model.parameters()])

    if pmin_ctx.prev_grad is None or pmin_ctx.prev_param is None:
        dtype = torch.float32  # Default dtype
        if ith_step != 0:
            logging.error(
                f"Previous gradients or parameters are None at batch index {ith_step}"
            )
    else:
        # L2 norms
        grad_norm = torch.linalg.norm(curr_grads, ord=2)
        param_norm = torch.linalg.norm(curr_params, ord=2)

        grad_diff = torch.linalg.norm(curr_grads - pmin_ctx.prev_grad, ord=2)
        param_diff = torch.linalg.norm(curr_params - pmin_ctx.prev_param, ord=2)

        if param_diff < EPSILON:
            logging.warning("Parameter update too small for stable Lipschitz estimate.")
            lipschitz = 1  # Or maintain previous estimate
        else:
            lipschitz = grad_diff / param_diff

        if lipschitz < EPSILON:
            logging.warning(
                "Lipschitz constant too small, possible numerical instability."
            )
            lipschitz = 1  # Or maintain previous estimate

        pmin = math.log2((4 + 3 * math.sqrt(2)) * lipschitz * param_norm / grad_norm)
        pmin_ctx.pmin_window[ith_step % WINDOW_SIZE] = pmin
        pmin_moving_avg = np.nanmean(pmin_ctx.pmin_window)

        if pmin_moving_avg <= 7:
            dtype = torch.float16
        elif pmin_moving_avg <= 10:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        logging.debug(f""" Metrics at batch {ith_step}:
Gradient diff: {grad_diff:.4f}
Parameter diff: {param_diff:.4f}
Lipschitz constant: {lipschitz:.4f}
Gradient norm: {grad_norm:.4f}
Parameter norm: {param_norm:.4f}
Estimated pmin: {pmin:.2f}
PMIN moving avg: {pmin_moving_avg:.2f}
Selected dtype: {dtype}""")

        current_dtype = next(model.parameters()).dtype
        if current_dtype != dtype:
            logging.info(f"Changing model precision to {dtype}")
            model = model.to(dtype=dtype)
            _sync_optimizer(optimizer, model=model, dtype=dtype)

    pmin_ctx.prev_grad = curr_grads.clone()
    pmin_ctx.prev_param = curr_params.clone()

    return dtype
