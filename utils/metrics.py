import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, auc, roc_curve


def get_accuracy(y_hat, y, se=False):
    accuracy = (y_hat.argmax(dim=1) == y).sum().item() * 1.0 / len(y)
    if se:
        se = np.sqrt(accuracy * (1 - accuracy) / len(y))
        return accuracy, se
    return accuracy


def get_f1(y_hat, y):
    return f1_score(y_hat.argmax(dim=1).detach().cpu(), y.detach().cpu(), average='macro')


def get_auc(y_hat, y):
    pred = y_hat.softmax(1).detach().cpu().numpy()
    target = y.detach().cpu().numpy()
    try:
        if y_hat.shape[1] > 2:
            auc = roc_auc_score(target, pred, multi_class='ovo', labels=range(pred.shape[1]))
        else:
            auc = roc_auc_score(target, pred[:, 1])
    except:
        auc = torch.tensor(0.0)
    return auc


def get_auc_label(y_true, model_outs, label):
    fpr, tpr, _ = roc_curve(y_true.numpy(), model_outs.numpy()[:, label], pos_label=label)
    return auc(fpr, tpr)
