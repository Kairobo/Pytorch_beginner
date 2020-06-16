import torch
import torch.nn.functional as F
import os
import numpy as np

def classify_loss_acc(batch_truth, batch_pred, use_cuda=False, loss_weight = None):
    if use_cuda:
        batch_truth = batch_truth.cuda()
    if loss_weight is None:
        loss = F.nll_loss(batch_pred, batch_truth)
    else:
        loss = F.nll_loss(batch_pred, batch_truth, weight=loss_weight, reduction='mean')
    pred = batch_pred.data.max(1, keepdim=True)[1]
    correct = pred.eq(batch_truth.data.view_as(pred)).sum()
    return loss, correct


def update_confusion_matrix(confusion_matrix, batch_truth, batch_pred):
    batch_size = len(batch_truth)
    for i in range(batch_size):
        truth_id = batch_truth[i]
        pred_id = batch_pred[i]
        confusion_matrix[truth_id,pred_id] += 1


def precision_from_confusion_matrix(confusion_matrix):
    num_classes = len(confusion_matrix)

    precision_list = []
    for i in range(num_classes):
        num_TP = confusion_matrix[i,i]

        num_FP_plus_TP = np.sum(confusion_matrix[:,i])
        if num_FP_plus_TP == 0:
            precision_list.append(-1)
        else:
            precision_list.append(num_TP / num_FP_plus_TP)

    return precision_list


def recall_from_confusion_matrix(confusion_matrix):
    num_classes = len(confusion_matrix)

    recall_list = []
    for i in range(num_classes):
        num_TP = confusion_matrix[i,i]

        num_FN_plus_TP = np.sum(confusion_matrix[i,:])
        if num_FN_plus_TP == 0:
            recall_list.append(-1)
        else:
            recall_list.append(num_TP / num_FN_plus_TP)

    return recall_list


def accuracy_from_confusion_matrix(confusion_matrix):
    num_correct = np.sum([confusion_matrix[i,i] for i in range(len(confusion_matrix))])

    num_sum = np.sum(confusion_matrix)
    if num_sum > 0:
        return num_correct / num_sum
    else:
        return -1



def load_model(model_name, model_path, network, optimizer=None, device = 'cpu'):
    model_setting_list = model_name.split('_')
    if len(model_setting_list) == 2:
        model_type, device_type = model_setting_list
    else:
        model_type, device_type, task = model_setting_list

    model_path = os.path.join(model_path, "model.pth")
    optim_path = os.path.join(model_path, "optimizer.pth")

    if device_type == 'gpu':
        if device == 'gpu':
            # load from gpu and use in gpu
            device = torch.device('cuda')
            network.load_state_dict(torch.load(model_path))
            network.to(device)
        else:
            # load from gpu and use in cpu
            device = torch.device('cpu')
            network.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # load from cpu and use in either cpu or gpu
        network.load_state_dict(torch.load(model_path))
    print("Load model from {}".format(model_path))

    # possibly load optimizer if training, no need to if testing
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(optim_path))
        print("Load optimizer from {}".format(optim_path))