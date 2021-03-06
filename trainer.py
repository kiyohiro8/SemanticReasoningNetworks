import numpy as np
import torch
import torch.nn.functional as F
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
        
        max_epoch = params["max_epoch"]
        learning_rate = params["learning_rate"]
        batch_size = params["batch_size"]
        max_char_length = params["max_char_length"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SRNModel(params)
        model.to(device)

        criterion = cal_performance

        train_dataset = ImgTxtDataset(image_dir, train_csv, max_char_length)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True, 
            num_workers=4)
        valid_dataset = ImgTxtDataset(
            image_dir, 
            valid_csv, 
            max_char_length, 
            train_dataset.char_encoder, 
            is_train=False)
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            drop_last=False)
        stop_char = train_dataset.stop_char_index

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
                pvam_out, gsrm_out, f_out = model(images)
                mask = texts.ne(stop_char)
                loss = criterion(pvam_out, gsrm_out, f_out, texts, mask)

                optimizer.zero_grad()
                loss.backword()
                optimizer.step()

                loss = loss.item()

            val_loss, val_acc = self.validation(model, valid_dataloader, stop_char, device)
            if val_loss < best_val_loss:
                print(f"epoch: {epoch}, val_loss: {val_loss}, val_acc: {val_acc}")
                best_val_loss = val_loss
                weights = {
                    "weights": model.state_dict(),
                    "opt_weights": optimizer.state_dict(),
                    "char_encoder": train_dataset.char_encoder,
                    "epoch": epoch,
                    "val_loss": best_val_loss
                }
                torch.save(weights, f"{epoch}.pth")

    def validation(self, model, dataloader, stop_char, device):

        n_pred = 0
        total_loss = 0
        total_correct = 0

        model.eval()
        for batch in dataloader:
            images, texts = batch
            n_pred += images.shape[0]
            images = images.to(device)
            texts = texts.to(device)
            pvam_out, gsrm_out, f_out = model(images)
            mask = texts.ne(stop_char)
            
            pvam_loss = cal_loss(pvam_out, texts, mask)
            gsrm_loss = cal_loss(gsrm_out, texts, mask)
            f_loss = cal_loss(f_out, texts, mask)
            total_loss += pvam_loss.mean().item() + gsrm_loss.mean().item() + f_loss.mean().item()
            
            #pvam_correct = pvam_out.eq(texts).masked_select(mask)
            #gsrm_correct = gsrm_out.eq(texts).masked_select(mask)
            f_correct = f_out.eq(texts).masked_select(mask)
            total_correct += f_correct.sum().item()       
        model.train()

        val_loss = total_loss / n_pred
        val_acc = total_correct / n_pred

        return val_loss, val_acc


def cal_performance(pvam_out, gsrm_out, f_out, gt, mask=None):
    ''' Apply label smoothing if needed '''

    weights = [1.0, 0.15, 2.0]

    pvam_out = pvam_out.view(-1, pvam_out.shape[-1])
    gsrm_out = gsrm_out.view(-1, gsrm_out.shape[-1])
    f_out = f_out.view(-1, f_out.shape[-1])

    if mask is not None:
        mask = mask.view(-1)
    pvam_loss = cal_loss(pvam_out, gt, mask)
    gsrm_loss = cal_loss(gsrm_out, gt, mask)
    f_loss = cal_loss(f_out, gt, mask)

    loss = weights[0] * pvam_loss + weights[1] * gsrm_loss + weights[2] * f_loss

    return loss


def cal_loss(pred, gold, mask=None):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if mask is not None:
        loss = F.cross_entropy(pred, gold, reduction='none')
        loss = loss.masked_select(mask)
        loss = loss.mean()
    else:
        loss = F.cross_entropy(pred, gold)

    return loss

