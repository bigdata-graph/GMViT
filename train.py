import torch
import torch.nn as nn
import os, shutil, json
import argparse
from GMViT import GMViT, SVCNN
from dataloader import MultiviewImgDataset

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import os

TRAIN_NAME = __file__.split('.')[0]


def parse_arguments():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='GMViT', metavar='N', help='Name of the experiment')

    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')

    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epochs', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--Tmax', type=int, default=50, metavar='N', help='Max iteration number of scheduler. ')
    parser.add_argument('--use_sgd', type=int, default=True, help='Use SGD')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate (default: 0.01, 0.1 if using sgd)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval', type=int, default=False, help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument("--num_models", type=int, help="number of models per class", default=0)
    parser.add_argument("--no_pretraining", dest='no_pretraining', action='store_true')
    parser.add_argument("--cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet18")
    parser.add_argument("--num_views", type=int, help="number of views", default=20)
    parser.add_argument("--group_num", type=int, help="number of views", default=12)
    parser.add_argument("--im_train_path", type=str, default="data/modelnet40v2png_ori4/*/train")
    parser.add_argument("--im_val_path", type=str, default="data/modelnet40v2png_ori4/*/test")
    
    return parser.parse_args()


def _init_(args):
    if args.name == '':
        args.name = TRAIN_NAME
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('models/' + args.name):
        os.makedirs('models/' + args.name)
    if not os.path.exists('models/' + args.name + '/' + 'models'):
        os.makedirs('models/' + args.name + '/' + 'models')
        

def CNN(args):
    pretraining = not args.no_pretraining
    cnet = SVCNN(nclasses=args.num_category, pretraining=args.no_pretraining, cnn_name=args.cnn_name)
    cnet.load_state_dict(torch.load('train_cnn/SVCNN_model/view-gcn/model-00000.pth'))

    return cnet


def train(args, model, io):
    torch.manual_seed(args.seed)
    if args.gpu_idx < 0:
        io.cprint('Using CPU')
    else:
        io.cprint('Using GPU: {}'.format(args.gpu_idx))
        torch.cuda.manual_seed(args.seed)

    train_dataset = MultiviewImgDataset(args.im_train_path, scale_aug=False, rot_aug=False, num_models=0, num_views=args.num_views,test_mode=True, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_dataset = MultiviewImgDataset(args.im_val_path, scale_aug=False, rot_aug=False, num_views=args.num_views,test_mode=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    print(model)
    for param in model.encoder_net_1.parameters():
        param.requries_grad = False
    model = model.to(device)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*10, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, 50, eta_min=args.lr)


    criterion = cal_loss
    epochs = args.epochs
    best_test_acc = 0
    for epoch in range(epochs):
        if epoch < args.Tmax:
            scheduler.step()
        elif epoch == args.Tmax:
            for param in model.parameters():
                param.requries_grad = True

        learning_rate = opt.param_groups[0]['lr']
    #     ####################
    #     # Train
    #     ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        step=0
        for i, data in enumerate(train_loader):
            N, V, C, H, W = data[1].size()
            img = data[1].view(-1, C, H, W).cuda()
            batch_size = N
            label = data[0].cuda().long()

            opt.zero_grad()
            logits = model(img)
            loss = criterion(logits, label)
            print('step:{}, train_loss:{}'.format(i, loss.item()))
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint('EPOCH #{}  lr = {}'.format(epoch, learning_rate))
        io.cprint(outstr)


        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        step=0
        for i, data in enumerate(test_loader):
            N, V, C, H, W = data[1].size()
            img = data[1].view(-1, C, H, W).cuda()
            # pc = pc.permute(0, 2, 1)
            batch_size = N
            label = data[0].cuda().long()

            with torch.no_grad():
                logits = model(img)
            loss = criterion(logits, label)
            print('step:{}, test_loss:{}'.format(i, loss.item()))
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss * 1.0 / count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'models/%s/models/model.t7' % args.name)
            io.cprint('Current best saved in: {}'.format('********** models/%s/models/model.t7 **********' % args.name))

    
def test(args, model, io):
    device = torch.device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))
    
    test_dataset = MultiviewImgDataset(args.im_val_path, scale_aug=False, rot_aug=False, num_views=args.num_views, test_mode=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    io.cprint('********** TEST STAGE **********')
    io.cprint('Reload best epoch:')
    
    #Try to load models
    model = model.to(device)
    model.load_state_dict(torch.load('models/GMViT/models/model.t7'))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for i, data in enumerate(test_loader):
        N, V, C, H, W = data[1].size()
        img = data[1].view(-1, C, H, W).cuda()
        batch_size = N
        label = data[0].cuda().long()

        with torch.no_grad():
            logits = model(img)
        print('step:{}'.format(i))
        preds = logits.max(dim=1)[1]
        count += batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test : test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    _init_(args)
    io = IOStream('models/' + args.name + '/train.log')
    io.cprint(str(args))
    model_img = CNN(args)
    model = GMViT(model=model_img, cnn_name=args.cnn_name, num_views=args.num_views, group_num=args.group_num)

    if not args.eval:
        train(args, model, io)
    else:
        test(args, model, io)

