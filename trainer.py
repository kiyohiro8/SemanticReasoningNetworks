import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import SRNModel
from dataset import ImgTxtDataset

class SRNTrainer(object):
    def __init__(self):
        pass

    def train(self, params):

        image_dir = params["image_dir"]
        train_csv = params["train_csv"]
        valid_csv = params["valid_csv"]

        learning_rate = params["learning_rate"]
        batch_size = params["batch_size"]
        max_char_length = params["max_char_length"]



        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SRNModel(params)
        model.to(device)

        criterion = cal_performance

        train_dataset = ImgTxtDataset(image_dir, train_csv, max_char_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=4)
        valid_dataset = ImgTxtDataset(
            image_dir, 
            valid_csv, 
            max_char_length, 
            train_dataset.char_encoder, 
            is_train=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False)

        optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.99))

        start_epoch = 1
        best_val_loss = np.inf
        for epoch in range(start_epoch, max_epoch + 1):
            print(f"epoch {epoch} start")
            for batch in train_dataloader:
                images, texts = batch
                preds = model(images)
                loss, _ = criterion(preds, texts)

                optimizer.zero_grad()
                loss.backword()
                optimizer.step()

                loss = loss.item()

            val_loss = self.validation(model, valid_dataloader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                weights = {
                    "weights": model.state_dict(),
                    "opt_weights": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss
                }
                torch.save(weights, f".pth")


def cal_performance(preds, gold, mask=None, smoothing='1'):
    ''' Apply label smoothing if needed '''

    loss = 0.
    n_correct = 0
    weights = [1.0, 0.15, 2.0]
    for ori_pred, weight in zip(preds, weights):
        pred = ori_pred.view(-1, ori_pred.shape[-1])
        # debug show
        t_gold = gold.view(ori_pred.shape[0], -1)
        t_pred_index = ori_pred.max(2)[1]

        mask = mask.view(-1)
        non_pad_mask = mask.ne(0) if mask is not None else None
        tloss = cal_loss(pred, gold, non_pad_mask, smoothing)
        if torch.isnan(tloss):
            print('have nan loss')
            continue
        else:
            loss += tloss * weight

        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item() if mask is not None else None

    return loss, n_correct




