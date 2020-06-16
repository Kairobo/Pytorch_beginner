import config
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
from dataset.custom_dataset import CustomDataset
from models.networks import *
from options import *
from utils.data_utils import *
from utils.ml_utils import *


# set the config manually
config_obj = config.obj

def train(args):
    # --- read the setting for training
    random_seed = args.randomSeed
    torch.backends.cudnn.enabled = args.useCuda
    torch.manual_seed(random_seed)

    n_epochs = args.numEpochs
    batch_size_train = args.batchSize
    log_interval = args.logInterval
    log_valinterval = args.logValInterval

    learning_rate = args.LR
    momentum = args.Momentum

    # data_config
    data_config = {}
    data_config['split'] = 'train'
    data_config["root_dir"] = args.customDataFolder
    data_config['input_resolution'] = args.inputResolution
    data_config['use_data_aug'] = args.useDataAug

    custom_data = CustomDataset(data_config=data_config)

    data_loader = DataLoader(custom_data, batch_size=batch_size_train, shuffle=True, num_workers=0)

    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    model_path = os.path.join(checkpoint_dir, args.modelName)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    network = TransferNet(num_classes=custom_data.get_num_of_classes())
    network = network.float()

    # optimizer
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, network.parameters()),
        lr=learning_rate, momentum=momentum
    )

    # load model if specified
    if args.useCuda:
        device = 'gpu'
    else:
        device = 'cpu'

    if args.loadModel:
        load_model(args.modelName, model_path, network, optimizer, device = device)

    if args.useCuda:
        network.cuda()

    eval_dict = test(args, network, custom_data)
    # train
    for epoch in range(n_epochs):
        train_corrects = 0
        train_losses = []
        for batch_idx, sample_batched in enumerate(data_loader):
            network.train()
            optimizer.zero_grad()
            if args.useCuda:
                sample_image = sample_batched['image'].float().cuda()
            else:
                sample_image = sample_batched['image'].float()

            output = network(sample_image)
            loss, correct = classify_loss_acc(sample_batched['label'], output,
                                                      use_cuda=args.useCuda)
            train_corrects += correct
            train_losses.append(loss.item())

            loss.backward()

            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_lv1: {:.6f}'.format(
                    epoch, batch_idx * len(sample_batched['image']), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader), loss.item()))

                train_losses.append(loss.item())


        if epoch % log_valinterval == 0:
            print('\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                np.mean(train_losses), train_corrects, len(data_loader.dataset),
                100. * train_corrects / len(data_loader.dataset)))

            eval_dict = test(args, network, custom_data)

            # save the model after each epoch
            torch.save(network.state_dict(), os.path.join(model_path, 'model.pth'))
            torch.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))

    print("training finished")

def test(args, network, custom_dataset):
    custom_dataset.test()
    num_classes = custom_dataset.get_num_of_classes()
    confusion_matrix = np.zeros((num_classes, num_classes))

    data_loader = DataLoader(custom_dataset, batch_size=4, shuffle=False, num_workers=0)

    network.eval()

    test_loss = 0
    test_losses = []
    correct = 0
    with torch.no_grad():
        for sample_batched in data_loader:
            if args.useCuda:
                output = network(sample_batched['image'].float().cuda())
                test_loss += F.nll_loss(output, sample_batched['label'].cuda(),
                                            size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(sample_batched['abel'].cuda().data.view_as(pred)).sum()
            else:
                output = network(sample_batched['image'].float())
                test_loss += F.nll_loss(output, sample_batched['label'], size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(sample_batched['label'].data.view_as(pred)).sum()

            # confusion matrix
            update_confusion_matrix(confusion_matrix,
                                    sample_batched['label'].cpu().numpy(), pred)

    test_loss /= len(data_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    # calculate the precision and the recall from the confusion matrix
    precision_list = precision_from_confusion_matrix(confusion_matrix)
    recall_list = recall_from_confusion_matrix(confusion_matrix)

    # dump evaluation result of test
    eval_dict = {}
    eval_dict['test'] = {}
    eval_dict['test']['confusion_matrix'] = confusion_matrix.tolist()
    eval_dict['test']['precision'] = precision_list
    eval_dict['test']['recall'] = recall_list
    eval_dict['test']['accuracy'] = accuracy_from_confusion_matrix(confusion_matrix)

    # save eval dictionary
    model_name = args.modelName
    model_path = os.path.join('./checkpoints/', model_name)
    eval_json_dir = os.path.join(model_path, 'eval_test.json')
    with open(eval_json_dir, 'w') as outfile:
        json.dump(eval_dict, outfile)

    print("test finished")

    return eval_dict

if __name__ == "__main__":
    args = parse_args()

    # set configs here, add other configs by adding to options.py or config.py
    args.customDataFolder = "/Users/kaijia/PycharmProjects/Pytorch_beginner/data"

    args.numEpochs = 3 # normally 3
    args.batchSize = 4

    args.useDataAug = False

    args.modelName = 'resnet_cpu'
    args.inputResolution = 256
    args.modelSaveInterval = 200

    train(args)



