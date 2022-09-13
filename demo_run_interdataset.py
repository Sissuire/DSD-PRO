""" 
DSD-PRO v2.0: add normalization (by default)
inter-dataset performance 
80%-20% for training/validation, and tested on other databases directly
+---------------------------
before this, you should have already done with the code of feature extraction  
+---------------------------
`CUDA_VISIBLE_DEVICES=1 nohup python3 -u demo_run_interdataset.py --database KoNViD > log.inter.KoNViD.out 2>&1 &`
"""

from argparse import ArgumentParser
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats

import sys
import pickle
import os


def read_info(base, file):
    name, mos = [], []
    with open(file, 'r') as f:
        for line in f:
            dis, score = line.split()
            name.append(base + dis)
            mos.append(float(score))
    name = np.stack(name)
    mos = np.stack(mos)
    return name, mos


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            # if param.dim() > 1:
            #     print(name, ': ', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            # else:
            #     print(name, ': ', num_param)
            total_param += num_param
    return total_param


class VQADataset(Dataset):
    def __init__(self, features, mos):
        super(VQADataset, self).__init__()

        self.n_data = len(mos)
        self.features = features
        self.mos = mos

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        return (self.features[idx], self.mos[idx])


class Embedding(nn.Module):
    def __init__(self, input_size, hidden_size=128, disable_norm=False):
        super(Embedding, self).__init__()

        norm_layer = nn.LayerNorm
        if disable_norm:
            norm_layer = nn.Identity

        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            norm_layer(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            # nn.Tanh()
        )

    def forward(self, x):
        return self.embedding(x)


class DSD_PRO(nn.Module):  
    """
        UGC-VQA based on Decomposition and Recomposition
        +-----------------------------------------------+
        version-2.0: add LayerNorm for a better generalization
    """
    def __init__(self, blk_list, hidden_size=32, disable_norm=False):

        super(DSD_PRO, self).__init__()
        reduce_size = hidden_size * 4

        norm_layer = nn.LayerNorm
        if disable_norm:
            norm_layer = nn.Identity
            print('disable norm layers')
        else:
            print('using Layer Normalization')
        self.blk1 = Embedding(blk_list[0], hidden_size=reduce_size, disable_norm=disable_norm)
        self.blk2 = Embedding(blk_list[1] + reduce_size, hidden_size=reduce_size, disable_norm=disable_norm)
        self.blk3 = Embedding(blk_list[2] + reduce_size, hidden_size=reduce_size, disable_norm=disable_norm)
        self.blk4 = Embedding(blk_list[3] + reduce_size, hidden_size=reduce_size, disable_norm=disable_norm)

        self.fc = nn.Sequential(
            nn.Linear(reduce_size, hidden_size),
            nn.ReLU(inplace=True),
            norm_layer(hidden_size),  
            nn.Linear(hidden_size, 1)
        )

        self.seg = [blk_list[0]]
        for i in range(1, len(blk_list)):
            self.seg.append(self.seg[i-1] + blk_list[i])

    def forward(self, x):
        batch, Ns = x.shape[0], x.shape[1]
        x = x.view(batch*Ns, -1)

        xf1 = x[:, :self.seg[0]]
        xf2 = x[:, self.seg[0]:self.seg[1]]
        xf3 = x[:, self.seg[1]:self.seg[2]]
        xf4 = x[:, self.seg[2]:]

        blk1 = self.blk1(xf1)

        blk2 = self.blk2(torch.cat([xf2, blk1], 1))
        blk2 += blk1

        blk3 = self.blk3(torch.cat([xf3, blk2], 1))
        blk3 += blk2

        blk4 = self.blk4(torch.cat([xf4, blk3], 1))
        blk4 += blk3

        score = self.fc(blk4)

        score = torch.mean(score.view(batch, Ns), dim=1, keepdim=True)
        return score


class DSD_FC(nn.Module):
    def __init__(self, input_size=4096*2, hidden_size=32):

        super(DSD_FC, self).__init__()
        self.hidden_size = hidden_size

        if hidden_size < input_size // 16 // 16:
            reduced_size = 512
        else:
            reduced_size = 128

        self.fc = nn.Sequential(
            nn.Linear(input_size, reduced_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(reduced_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, inputx):
        batch, Ns = inputx.shape[0], inputx.shape[1]
        inputx = inputx.view(batch*Ns, -1)

        score = self.fc(inputx)
        
        score = torch.mean(score.view(batch, Ns), dim=1)
        return score


if __name__ == "__main__":
    parser = ArgumentParser(description='UGC-VQA based on Decomposition and Recomposition (inter-dataset)')

    parser.add_argument("--seed", type=int, default=12318,
                        help='seed for reproducibility (default: 12318)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--database', default='KoNViD', type=str,
                        help='database name (default: KoNViD)')
    parser.add_argument('--repeat_times', default=30, type=int,
                        help='the times of repeat running (default: 30)')

    parser.add_argument('--model', default='dsd', type=str,
                        help='model name (default: dsd)')

    parser.add_argument('--fc', action='store_true',
                        help='use FC layers rather than PRO')

    parser.add_argument('--disable_norm', action='store_true',
                        help='disable norm layers (version-1.0)')

    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val splits (default: 0)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='val ratio (default: 0.2)')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--nt', type=int, default=20,
                        help='[nt] frames to be sampled per video (default: 20 / [10, 20, 40])')

    args = parser.parse_args()

    # [DEBUG MODE]
    if sys.gettrace():

        print('in DEBUG mode')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        # args.epochs = 10  # only for simple DEBUG
    else:
        print('in RELEASE Mode')

    print(args)

    datasets = ['KoNViD', 'LIVE-VQC', 'YouTube']  # all your datasets

    database = args.database
    exp_id = args.exp_id

    path_ = '/mnt/disk/yongxu_liu/datasets/in_the_wild/'

    mos = {}
    for dset in datasets:
        base_path = path_ + dset + '/'
        info_file = base_path + dset + '_info.txt'
        _, mos[dset] = read_info(base_path, info_file)

    num_samples_all = {'KoNViD': 1200, 'LIVE-VQC': 585, 'YouTube': 1195}
    num_samples = num_samples_all[database]
    num_repeat = args.repeat_times
    index_all = np.zeros((num_repeat, num_samples), dtype=np.int)
    for ii in range(num_repeat):
        index_current = np.asarray(range(num_samples_all[database]))
        random.Random(ii * 123).shuffle(index_current)   # shuffle with certain seed
        index_all[ii] = index_current
    np.savetxt('rand_index_{}.txt'.format(database), index_all, fmt='%d')

    nT = args.nt

    # ------------ load feature ---------------------
    features = {}
    print('loading features ..')
    for dset in ['KoNViD', 'LIVE-VQC', 'YouTube']:

        with open('./data/{}.{}.NT-{}.pkl'.format('dsd_pro_appearance', dset, nT), 'rb') as f:
            features[dset] = pickle.load(f)     # [N, nt, 192*2+320*2+1088*2+1536*2]
        blk_list = [192 * 2, 320 * 2, 1088 * 2, 1536 * 2]

        sf_features = []
        with open('./data/{}.{}.NT-{}.pkl'.format('dsd_pro_motion', dset, nT), 'rb') as f:
            sf_features = pickle.load(f)  # [N, nt, 32*2+64*2+128*2+256*2]

        features[dset] = np.concatenate([features[dset][:, :, :192*2], sf_features[:, :, :32*2],
                                         features[dset][:, :, 192*2:(192+320)*2], sf_features[:, :, 32*2:(32+64)*2],
                                         features[dset][:, :, (192+320)*2:-1536*2], sf_features[:, :, (32+64)*2:-256*2],
                                         features[dset][:, :, -1536*2:], sf_features[:, :, -256*2:]], 2)
    blk_list = [(192+32) * 2, (320+64) * 2, (1088+128) * 2, (1536+256) * 2]
    input_size = (192+32+320+64+1088+128+1536+256) * 2

    # ------------------ fix seed -----------------------
    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    # ---------------------------------------------------

    torch.utils.backcompat.broadcast_warning.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perf = []
    for t in range(num_repeat):
        index = index_all[t]
        
        pos_train_end = int((1 - args.val_ratio) * num_samples)
        trainindex = index[:pos_train_end]   # the first 80%
        evalindex = index[pos_train_end:]    # the last 20%

        trainindex.sort()
        evalindex.sort()

        # scale = mos.max()  # normalized to [0.1, 0.95]
        train_dataset = VQADataset(features[database][trainindex], mos[database][trainindex])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True, drop_last=True)
        val_dataset = VQADataset(features[database][evalindex], mos[database][evalindex])
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset)
        
        test_dataset, test_loader = {}, {}
        for dset in datasets:
            if dset == database:
                continue
            test_dataset[dset] = VQADataset(features[dset], mos[dset])
            test_loader[dset] = torch.utils.data.DataLoader(dataset=test_dataset[dset])

        num_train = len(train_dataset)
        num_eval = len(val_dataset)

        if args.fc:
            print('building Full-Connected layers for fusion ..')
            model_prefix = '{}_fc_inter'.format(args.model)
            model = DSD_FC(input_size=input_size).to(device)
        else:
            print('building Recursive layers for fusion ..')
            model_prefix = '{}_pro_inter'.format(args.model)
            model = DSD_PRO(blk_list=blk_list, disable_norm=args.disable_norm).to(device)

        print('Total trainable parameters: {}'.format(count_parameters(model)))
        print('Training/Validation: {}/{}'.format(num_train, num_eval))

        if not os.path.exists('models_{}'.format(model_prefix)):
            os.makedirs('models_{}'.format(model_prefix))
        trained_model_file = 'models_{}/{}-EXP{}'.format(model_prefix, args.database, exp_id)
        if not os.path.exists('results_{}'.format(model_prefix)):
            os.makedirs('results_{}'.format(model_prefix))
        save_result_file = 'results_{}/{}-EXP{}'.format(model_prefix, args.database, exp_id)

        print('EXP ID: {}, in {}, with {}'.format(exp_id, database, model_prefix))

        criterion = nn.L1Loss()  # L1 loss
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val_criterion = -1  # SROCC min
        SROCC, KROCC, PLCC, RMSE = 0, 0, 0, 0
        for epoch in range(args.epochs):
            # Train
            model.train()
            for i, (feat, label) in enumerate(train_loader):
                feat = feat.to(device).float()
                label = label.to(device).float()
                optimizer.zero_grad()  #

                outputs = model(feat)
                loss = criterion(outputs.view(label.shape), label)
                loss.backward()
                optimizer.step()

            # Eval
            model.eval()
            y_eval_pred, y_eval_label = [], []
            with torch.no_grad():
                for i, (feat, label) in enumerate(val_loader):
                    feat = feat.to(device).float()
                    outputs = model(feat)

                    y_eval_pred.extend(outputs.cpu().float().numpy())
                    y_eval_label.extend(label.cpu().float().numpy())
            y_eval_pred = np.asarray(y_eval_pred).reshape(-1)
            y_eval_label = np.asarray(y_eval_label).reshape(-1)

            val_PLCC = stats.pearsonr(y_eval_pred, y_eval_label)[0]   # we do NOT use nonlinear regression for simplicity
            val_SROCC = stats.spearmanr(y_eval_pred, y_eval_label)[0]
            val_RMSE = np.sqrt(((y_eval_pred - y_eval_label) ** 2).mean())
            val_KROCC = stats.stats.kendalltau(y_eval_pred, y_eval_label)[0]

            # Update the model with the best val_SROCC
            if val_SROCC > best_val_criterion:
                print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(exp_id, epoch))
                print("Val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(val_SROCC, val_KROCC, val_PLCC, val_RMSE))

                torch.save(model.state_dict(), trained_model_file)
                best_val_criterion = val_SROCC  # update best val SROCC

        # Test
        model.load_state_dict(torch.load(trained_model_file))
        model.eval()

        with torch.no_grad():
            this_perf = []
            for dset in test_loader.keys():

                y_test_pred, y_test_label = [], []
                for i, (feat, label) in enumerate(test_loader[dset]):
                    feat = feat.to(device).float()
                    outputs = model(feat)

                    y_test_pred.append(outputs.cpu().float().numpy())
                    y_test_label.append(label.cpu().float().numpy())

                y_test_pred = np.stack(y_test_pred).reshape(-1)
                y_test_label = np.stack(y_test_label).reshape(-1)

                test_PLCC = stats.pearsonr(y_test_pred, y_test_label)[0]   # we do NOT use nonlinear regression for simplicity
                test_SROCC = stats.spearmanr(y_test_pred, y_test_label)[0]
                test_RMSE = np.sqrt(((y_test_pred - y_test_label) ** 2).mean())
                test_KROCC = stats.stats.kendalltau(y_test_pred, y_test_label)[0]

                print('-----' * 5)
                print("Test on {}: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(dset, test_SROCC, test_KROCC, test_PLCC, test_RMSE))

                this_perf.extend([test_SROCC, test_KROCC, test_PLCC, test_RMSE])
                np.save('results_{}/train_on_{}_test_on_{}-EXP{}'.format(
                    model_prefix, args.database, dset, exp_id),
                    np.concatenate([y_test_pred, y_test_label, 
                    np.asarray([test_SROCC, test_KROCC, test_PLCC, test_RMSE]).reshape(-1)], 0))

            print('-----' * 5)
            perf.append(this_perf)
            exp_id += 1

    print('========' * 10)
    for x_ in test_loader.keys():
        print(' {} /'.format(x_), end='', flush=True)
    print('\n------------------')
    print('_MEAN_ performance: SROCC/KROCC/PLCC/RMSE: {}'.format(np.mean(perf, 0)))
    print('MEDIAN performance: SROCC/KROCC/PLCC/RMSE: {}'.format(np.median(perf, 0)))
    print('_STDEV performance: SROCC/KROCC/PLCC/RMSE: {}'.format(np.std(perf, 0)))
    print('-------' * 10)
