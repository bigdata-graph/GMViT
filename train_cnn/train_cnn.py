import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import SingleImgDataset
from model.SVCNN import SVCNN
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="SVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=400)
parser.add_argument("-num_models", type=int, help="number of models per class", default=0)
parser.add_argument("-lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.001)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet18")
parser.add_argument("-num_views", type=int, help="number of views", default=20)
parser.add_argument("-train_path", type=str, default="/21085401045/Datasets/modelnet40v2png_ori4/modelnet40v2png_ori4/*/train")
parser.add_argument("-val_path", type=str, default="/21085401045/Datasets/modelnet40v2png_ori4/modelnet40v2png_ori4/*/test")

parser.set_defaults(train=False)

def create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    seed_torch()
    args = parser.parse_args()
    pretraining = not args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()
    log_dir = args.name+'_model'
    create_folder(log_dir)
    cnet = SVCNN(args.name, nclasses=40, pretraining=True, cnn_name=args.cnn_name)

    optimizer = optim.SGD(cnet.parameters(), lr=1e-1, weight_decay=args.weight_decay, momentum=0.9)
    n_models_train = args.num_models*args.num_views
    
    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=400, shuffle=True, num_workers=4)
    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=400, shuffle=False, num_workers=4)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
    trainer.train(30)
    # cnet.load_state_dict(torch.load('SVCNN_model/view-gcn/model-00000.pth'))
    # trainer.update_validation_accuracy(1)
