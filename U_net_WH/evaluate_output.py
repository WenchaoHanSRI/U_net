import torch
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn import metrics
import wandb

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate_output(net, dataloader, device, exp, logimage):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score, precision, recall, f1, accuracy = 0, 0, 0, 0, 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = mask_true // 255
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                # dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                #                                     reduce_batch_first=False)
                dice = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                dice_score += dice
                m_p = mask_pred[:, 1:, ...].cpu().numpy()
                m_t = mask_true[:, 1:, ...].cpu().numpy()
                m_p = m_p.flatten()
                m_t = m_t.flatten()
                precision += metrics.precision_score(m_t, m_p)
                recall += metrics.recall_score(m_t, m_p)
                f1 += metrics.f1_score(m_t, m_p)
                accuracy += metrics.accuracy_score(m_t, m_p)

                exp.log({
                    'Dice': dice,
                    'precision': metrics.precision_score(m_t, m_p),
                    'recall': metrics.recall_score(m_t, m_p),
                    'f1': metrics.f1_score(m_t, m_p),
                    'accuracy': metrics.accuracy_score(m_t, m_p)
                })


        if logimage == True:
            exp.log({
                'images': wandb.Image(image[0].cpu()),
                'masks': {
                    'true': wandb.Image(mask_true.cpu().squeeze()[1]),
                    'pred': wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu()),
                }
            })

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return mask_pred, mask_true, image, f1 / num_val_batches,\
           precision / num_val_batches, recall / num_val_batches,\
           accuracy / num_val_batches, dice_score / num_val_batches
