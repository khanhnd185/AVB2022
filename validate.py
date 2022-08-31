import os
import pickle
import pandas as pd
from dataset import AVBFeature, AVBWav, AVBH5py
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from sam import SAM
from helpers import *
from model import AvbWav2vecLstm, AvbWav2vec, AvbWav2vecFeatureLstm

def val(net, validldr, criteria, metric, task):
    total_losses = AverageMeter()
    yhat = {}
    net.eval()
    all_y = None
    all_yhat = None

    for batch_idx, (x, y, leng) in enumerate(tqdm(validldr)):
        with torch.no_grad():
            x = x.float()
            y = y.float()
            x = x.cuda()
            y = y.cuda()
            yhat = net(x, leng)
            loss = criteria(yhat, y)
            total_losses.update(loss.data.item(), x.size(0))

            if all_y == None:
                all_y = y.clone()
                all_yhat = yhat.clone()
            else:
                all_y = torch.cat((all_y, y), 0)
                all_yhat = torch.cat((all_yhat, yhat), 0)
    all_y = all_y.cpu().numpy()
    all_yhat = all_yhat.cpu().numpy()
    metrics = metric(all_y, all_yhat)
    return total_losses.avg(), metrics

def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--net', '-n', default='AvbWav2vecLstm', help='Net name')
    parser.add_argument('--input', '-i', default='', help='Input file')
    parser.add_argument('--task', '-t', default='two', help='Task')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--layer', '-l', type=int, default=12, help='Number of encoder layers')
    parser.add_argument('--datadir', '-d', default='../../../Data/A-VB/', help='Data folder path')

    args = parser.parse_args()
    task = args.task
    resume = args.input
    net_name = args.net
    data_dir = args.datadir
    batch_size = args.batch
    num_layers = args.layer

    annotation_file = os.path.join(data_dir, 'labels', task + '_info.csv')
    wav_path = os.path.join(data_dir, 'audio', 'wav')
    h5p_path = '../competitions/A-VB2022/end-to-end_based/' + task + '/data_wav2vec'

    if task == 'type':
        metric  = AvgCCC
    else:
        if net_name == 'AvbWav2vecFeatureLstm':
            validset = AVBH5py(annotation_file, h5p_path, 'Val')
        else:
            validset = AVBWav(annotation_file, wav_path, 'Val')
        criteria = CCCLoss()
        metric  = AvgCCC

        if task == 'high':
            num_output = 10
        elif task == 'two':
            num_output = 2
        elif task == 'culture':
            num_output = 40

    validldr = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate)

    if net_name == 'AvbWav2vec':
        net = AvbWav2vec(num_output, freeze_extractor=True, layer=num_layers)
    elif net_name == 'AvbWav2vecFeatureLstm':
        net = AvbWav2vecFeatureLstm(num_output)
    else:
        net = AvbWav2vecLstm(num_output, freeze_extractor=True, layer=num_layers)
    if resume != '':
        print("Resume form | {} ]".format(resume))
        net = load_state_dict(net, resume)

    net = nn.DataParallel(net).cuda()
    val_loss, val_metrics = val(net, validldr, criteria, metric, task)

    infostr = {'Task {}: {:.5f},{:.5f}'
            .format(task,
                    val_loss,
                    val_metrics)}
    print(infostr)


if __name__=="__main__":
    main()
