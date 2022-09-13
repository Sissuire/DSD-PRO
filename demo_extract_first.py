"""
feature extraction for DSD-PRO
+---------------------------
`CUDA_VISIBLE_DEVICES=1 nohup python3 -u demo_extract_first.py --database KoNViD --seg 20 > log.extract.KoNViD.nt-20.out 2>&1 &`
"""
import os
import cv2
import sys
import torch
import pickle
import numpy as np
from PIL import Image
from model_zoo.InceptionResNet_v2 import koncept512
from model_zoo.slowfast_network import sfresnet50
import torchvision.transforms as tf
from argparse import ArgumentParser


def get_info(dataset='KoNViD', basep='/mnt/disk/yongxu_liu/datasets/in_the_wild/'):
    base = '{}/{}'.format(basep, dataset)
    info_file = '{}/{}_info.txt'.format(base, dataset)

    name, mos = [], []
    with open(info_file, 'r') as f:
        for line in f:
            dis, score = line.split()
            name.append('{}/{}'.format(base, dis))
            mos.append(float(score))
    name = np.stack(name)
    mos = np.stack(mos)

    return name


if __name__ == '__main__':

    parser = ArgumentParser(description='feature extraction for DSD-PRO')
    parser.add_argument('--database', default='KoNViD', type=str,
                        help='database name (default: KoNViD)')
    parser.add_argument('--seg', default=20, type=int,
                        help='the number of segments to use (default: 20)')
    args = parser.parse_args()
    print(args)

    # [DEBUG MODE]
    if sys.gettrace():
        print('in DEBUG mode')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        args.database = 'LIVE-VQC'
    else:
        print('in RELEASE Mode')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    model_iqa = koncept512()
    print('initializing quality-aware model ..')
    model_iqa.load_state_dict(torch.load('./pretrained_model/KonCept512.pth', map_location='cpu'))
    model_iqa = model_iqa.to(device)
    model_iqa.eval()

    model_mot = sfresnet50(num_classes=600)
    print('initializing motion-aware model ..')
    checkpoint = torch.load('./pretrained_model/slowfast_k600_50_224.pth.tar', map_location='cpu')
    state_dict = checkpoint['model']
    model_state = {}
    for key in state_dict.keys():
        if 'module.' in key:
            model_state[key[7:]] = state_dict[key]
        else:
            model_state[key] = state_dict[key]
    model_mot.load_state_dict(model_state)
    model_mot = model_mot.to(device)
    model_mot.eval()
    # -----------------------

    nt = args.seg
    dataset = args.database
    dataset_file = get_info(dataset=dataset)

    transform_iqa = tf.Compose([
        tf.ToTensor(),
        tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_mot = tf.Compose([
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    feat_iqa = torch.zeros((len(dataset_file), nt, 6272), dtype=torch.float)  # 192*2 + 320*2 + 1088*2 + 1536*2
    feat_mot = torch.zeros((len(dataset_file), nt, 960), dtype=torch.float)  # 32*2 + 64*2 + 128*2 + 256*2
    with torch.no_grad():
        for i, file in enumerate(dataset_file):

            print('{:04d}, {}'.format(i, file))

            vidcap = cv2.VideoCapture(file)
            vid_len = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))

            flag_rescale = False
            if vid_width > vid_height > 1080 and vid_width > 1920:
                vid_width, vid_height = 1920, 1080
                flag_rescale = True
            elif vid_height > vid_width > 1080 and vid_height > 1920:
                vid_height, vid_width = 1920, 1080
                flag_rescale = True

            data = []
            ret, vframe = vidcap.read()
            frame_idx = 0
            while ret:
                vframe = Image.fromarray(cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB))
                if flag_rescale:
                    vframe = vframe.resize((vid_width, vid_height))
                data.append(vframe)

                ret, vframe = vidcap.read()
                frame_idx += 1

            vid_len = frame_idx

            seq = np.asarray(range(nt)) * (vid_len - 18) / (nt-1) + 5  # ignore the first and the last 5 frames
            seq = seq.astype('int')

            output_iqa = torch.Tensor()
            output_mot = torch.Tensor()

            for frame_idx in seq:

                input_iqa = torch.zeros((1, 3, vid_height, vid_width), dtype=torch.float)
                input_mot = torch.zeros((1, 3, 8, vid_height, vid_width), dtype=torch.float)

                input_iqa[0] = transform_iqa(data[frame_idx + 4])
                for iii in range(8):
                    input_mot[0, :, iii] = transform_mot(data[frame_idx + iii])

                # ---------------------------------------------------------------
                input_iqa = input_iqa.to(device)
                input_mot = input_mot.to(device)
                f_iqa = model_iqa(input_iqa)
                f_mot = model_mot(input_mot)

                output_iqa = torch.cat((output_iqa, f_iqa.cpu()), 0)
                output_mot = torch.cat((output_mot, f_mot.cpu()), 0)

            feat_iqa[i] = output_iqa
            feat_mot[i] = output_mot

        if not os.path.exists('./data/'):
            os.mkdir('./data/')
        with open('./data/{}.{}.NT-{}.pkl'.format('dsd_pro_appearance', dataset, nt), 'wb') as f:
            pickle.dump(feat_iqa, f)
        with open('./data/{}.{}.NT-{}.pkl'.format('dsd_pro_motion', dataset, nt), 'wb') as f:
            pickle.dump(feat_mot, f)
        print('done')
