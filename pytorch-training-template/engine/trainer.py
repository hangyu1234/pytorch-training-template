import torch


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device="cpu",
):
    """
    Train model for one epoch.
    """
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        # move data to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # clear old gradients
        optimizer.zero_grad()
        # forward
        prediction = model(batch_x)
        # calculate loss
        loss = loss_fn(
            prediction,
            batch_y,
        )
        # backward
        loss.backward()
        # update parameters
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

def evaluate(
    model,
    dataloader,
    loss_fn,
    device="cpu",
):
    """
    Evaluate model.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(batch_x)
            loss = loss_fn(
                prediction,
                batch_y,
            )
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss