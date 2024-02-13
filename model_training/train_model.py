import torch
import gc
import logging
import wandb
import os

logger = logging.getLogger(__name__)

def log_results(results, do_wandb_logging, e):
    if do_wandb_logging:
        wandb.log(results, step=e, commit=True)
    logger.info(f"Epoch {e}: {results}")

def store_model(model, savename):
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    torch.save(model.state_dict(), savename)

def train_model(
    model: torch.nn.Module, 
    model_name: str,
    dl_train: torch.utils.data.DataLoader,
    dl_val_dict: dict,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    eval_every: int,
    store_every: int,
    device: str,
    model_savedir: str,
    do_wandb_logging: bool,
    start_epoch: int=0,
    percentage_batches: float=1.0
    ):
    """
    Train model for given parameters.

    Args:
        model (torch.nn.Module): Model to be trained
        model_name (str): name of model type
        dl_train (torch.utils.data.DataLoader): DataLoader for training data
        dl_val_dict (dict): dictionary with DataLoaders for validation data
        criterion (torch.nn.Module): Loss function to be optimized
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): LR Scheduler
        num_epochs (int): Number of training epochs
        eval_every (int): Evaluate model with validation data every n epochs.
        store_every (int): Store model weights every n epochs.
        device (str): Device to be trained on ('cuda'/'cpu')
        model_savedir (str): Directory where model is stored.
        do_wandb_logging (bool): boolean specifying whether results are logged to weights and biases
        start_epoch (int): epoch to start with (default: 0)
        percentage_batches (float): fraction of batches to be considered (default: 1.0)
    """

    # Compute metrics before first iteration
    metrics_epoch = run_one_epoch(model, dl_train, criterion, optimizer, device, update_params=False)
    metrics_epoch = {f"train_{key}": val for key, val in metrics_epoch.items()}
    for val_name, dl_val in dl_val_dict.items():
        metrics_val = run_one_epoch(model, dl_val, criterion, optimizer, device, update_params=False)
        metrics_val = {f"{val_name}_{key}": val for key, val in metrics_val.items()}
        metrics_epoch = {**metrics_epoch, **metrics_val}
    log_results(metrics_epoch, do_wandb_logging, start_epoch)

    for epoch in range(start_epoch+1, start_epoch+num_epochs+1):
        metrics_epoch = run_one_epoch(model, dl_train, criterion, optimizer, device, update_params=True, percentage_batches=percentage_batches)
        metrics_epoch = {f"train_{key}": val for key, val in metrics_epoch.items()}

        if epoch % eval_every == 0:
            for val_name, dl_val in dl_val_dict.items():
                logger.info(f"Running evaluation for {val_name}")
                metrics_val = run_one_epoch(model, dl_val, criterion, optimizer, device, update_params=False)
                metrics_val = {f"{val_name}_{key}": val for key, val in metrics_val.items()}
                metrics_epoch = {**metrics_epoch, **metrics_val}
        
        log_results(metrics_epoch, do_wandb_logging, epoch)
        
        if epoch % store_every == 0:
            store_model(model, f"{model_savedir}/checkpoint_{model_name}_{epoch}.pth")

        if scheduler:
            scheduler.step()

    store_model(model, f"{model_savedir}/checkpoint_{model_name}_last.pth")

def run_one_epoch(
    model: torch.nn.Module, 
    dl: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    update_params: bool,
    percentage_batches: float=1.0
    ):
    """
    Runs the model for one epoch, either for training or validation.

    Args:
        model (torch.nn.Module): Model to be trained
        dl (torch.utils.data.DataLoader): DataLoader with training/validation data.
        criterion (torch.nn.Module): Loss function to be optimized
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to be trained on ('cuda'/'cpu')
        update_params (bool): Boolean specifying wether models weights are to be updated (training) or not (validation)
        percentage_batches (float): fraction of batches to be considered (default: 1.0)

    Returns:
        dict: Dictionary with metrics to be logged
    """

    model.to(device)
    running_loss = torch.tensor(0).float()

    model.train() if update_params else model.eval()

    y_true = []
    model_outs = []

    cancel_after = int(len(dl) * percentage_batches)
    log_every = len(dl) // 20
    for i, (imgs, labels) in enumerate(dl):
        if i % log_every == 0:
            logger.info(f"Batch {i+1}/{len(dl)}")
        if i > cancel_after:
            break
        if update_params:
            optimizer.zero_grad()

        imgs = imgs.to(device)
        outputs = model(imgs)
        outputs= outputs.cpu()
        loss = criterion(outputs, labels)

        if update_params:
            loss.backward()
            optimizer.step()
        
        model_outs.append(outputs.detach())
        y_true.append(labels)

        running_loss += loss.data.clone().cpu()

    y_true = torch.cat(y_true)
    model_outs = torch.cat(model_outs)

    y_hat = model_outs.argmax(1)

    results = {
        'loss': running_loss.item() / len(dl.dataset),
        'accuracy': (y_true == y_hat).numpy().mean(),
        'mean_logit': model_outs.mean().numpy(),
        'min_logit': model_outs.min().numpy(),
        'max_logit': model_outs.max().numpy()
        }

    model.cpu()
    torch.cuda.empty_cache(); gc.collect()

    return results

