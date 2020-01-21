from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
from tqdm import autonotebook, tqdm_notebook as tqdm


def train(model, loss_fn, n_epochs, patience, batch_size, lr, exp_id, num_lines=None):
    writer = SummaryWriter(os.path.join(OBSERVER_PATH, exp_id), str(model))
    last_model_path = os.path.join(OBSERVER_PATH, exp_id, "test_last_model.pth") #saves the last model during the training
    best_model_path = os.path.join(OBSERVER_PATH, exp_id, "test_best_model.pth") #saves the best model during the training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        train_dataset = Dataset(num_lines, train=True) #dataset to be used, depends on the case
        val_dataset = Dataset(num_lines)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size, 
                                      collate_fn=pad_batch, 
                                      worker_init_fn=set_seeds,
                                      )
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=batch_size, 
                                    collate_fn=pad_batch, 
                                    worker_init_fn=set_seeds)

        optimizer = RAdam(model.parameters(), lr=lr) #RAdam optimizer, can be used any RAdam from different libraries

        epochs = range(1, n_epochs + 1)
        best_loss = np.inf
        epochs_since_best = 0

        for epoch in epochs:
            epoch_dataloader = train_dataloader


            train_info = run_epoch("train", epoch, model, epoch_dataloader, loss_fn, optimizer, batch_size,
                                   extra_metrics=[accuracy], writer=writer, device=device)
            val_info = run_epoch("validation", epoch, model, val_dataloader, loss_fn, optimizer, batch_size,
                                 extra_metrics=[accuracy], writer=writer, device=device)

            if val_info["loss"] < best_loss:
                best_loss = val_info["loss"]
                torch.save(model, best_model_path)
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            torch.save(model, last_model_path, pickle_module=dill)
            if epochs_since_best > patience:
                break

            writer.commit()
    except Exception as e:
        print(e)
        torch.cuda.empty_cache()  
    

from functools import partial
WMA_WINDOW = 1000

def run_epoch(phase, epoch, model, dataloader, loss_fn, optimizer, batch_size, scheduler=None, extra_metrics=None, device=None,
              writer=None):
    training = phase == "train"
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}") # Progress bar to check the update of the model

    if extra_metrics is None:
        extra_metrics = {}
    elif isinstance(extra_metrics, list):
        extra_metrics_tmp = {}
        for metric in extra_metrics:
            if isinstance(metric, tuple):
                metric_name, metric_fn = metric
                extra_metrics_tmp[metric_name] = metric_fn
            else:
                extra_metrics_tmp[metric.__name__] = metric
        extra_metrics = extra_metrics_tmp

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)

    if training:
        model.train()
        mean_fn = partial(numpy_ewma_vectorized, window=int(WMA_WINDOW/batch_size + 1)) #calculate the mean function
    else:
        model.eval()
        mean_fn = np.mean

    losses = []
    extra_metrics_values = {metric_name: [] for metric_name, _ in extra_metrics.items()}
    for i, data in enumerate(progress_bar):
        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            elements_data, targets = data#
            results = model(elements_data[0],elements_data[1])#
            if results is None:
                continue
            loss = loss_fn(results, targets, elements_data[1])#
            if training:
                if not torch.isnan(loss):
                    loss.backward()#
                optimizer.step()#

                if scheduler is not None:
                    scheduler.step()
            losses.append(loss.item())#
            progress_bar_dict = {f"{phase} loss": mean_fn(losses)}
            for extra_metric_name, extra_metric_fn in extra_metrics.items():
                value_tmp = extra_metric_fn(results, targets.squeeze(1))#
                extra_metrics_values[extra_metric_name].append(value_tmp)
                progress_bar_dict[f"{phase} {extra_metric_name}"] =mean_fn(extra_metrics_values[extra_metric_name])
            progress_bar.set_postfix(progress_bar_dict)
    
    mean_loss = mean_fn(losses)

    info = {"loss": mean_loss}
    for extra_metric_name, _ in extra_metrics.items():
        info[extra_metric_name] = mean_fn(extra_metrics_values[extra_metric_name])

    if writer is not None:
        for metric_name, metric_mean_value in info.items():
            writer.add_scalar(f"{phase}.{metric_name}", metric_mean_value, epoch)

    return info