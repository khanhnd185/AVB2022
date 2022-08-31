import os
import pickle
import pandas as pd
from dataset import AVBFeature, AVBWav, AVBH5py, AVBWavType
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from sam import SAM
from helpers import *
from model import AvbWav2vecLstm, AvbWav2vec, AvbWav2vecFeatureLstm
import torchaudio

def train(net, trainldr, optimizer, epoch, epochs, learning_rate, criteria, task):
    total_losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)

    for batch_idx, (x, y, leng) in enumerate(tqdm(trainldr)):
        #adjust_learning_rate(optimizer, epoch, epochs, learning_rate, batch_idx, train_loader_len)
        x = x.float()
        y = y.float()
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        yhat = net(x, leng)
        loss = criteria(yhat, y)
        loss.backward()
        optimizer.step()
        total_losses.update(loss.data.item(), x.size(0))

    return total_losses.avg()

def train_sam(net, trainldr, optimizer, epoch, epochs, learning_rate, criteria, task):
    total_losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)

    for batch_idx, (x, y, leng) in enumerate(tqdm(trainldr)):
        mask = torch.ones(x.shape[0])
        mask = mask.float()
        mask = mask.cuda()
        # adjust_learning_rate(optimizer, epoch, epochs, learning_rate, batch_idx, train_loader_len)
        x = x.float()
        y = y.float()
        x = x.cuda()
        y = y.cuda()

        yhat = net(x, leng)
        loss = criteria(yhat, y, mask)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        yhat = net(x, leng)
        criteria(yhat, y, mask).backward()
        optimizer.second_step(zero_grad=True)

        total_losses.update(loss.data.item(), x.size(0))
    return total_losses.avg()

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
    parser.add_argument('--loss', '-L', default='ccc', help='Loss function')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--layer', '-l', type=int, default=12, help='Number of encoder layers')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of epoches')
    parser.add_argument('--lr', '-a', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--datadir', '-d', default='../../../Data/A-VB/', help='Data folder path')
    parser.add_argument('--outdir', '-o', default='train', help='output folder path')
    parser.add_argument('--wav', '-w', default='WAV2VEC2_BASE', help='Wav2vec version')
    parser.add_argument('--sam', '-s', action='store_true', help='Apply SAM optimizer')

    args = parser.parse_args()
    task = args.task
    loss = args.loss
    epochs = args.epoch
    resume = args.input
    wav2vec_name = args.wav
    net_name = args.net
    data_dir = args.datadir
    output_dir = args.outdir
    batch_size = args.batch
    learning_rate = args.lr
    num_layers = args.layer

    if wav2vec_name == 'WAV2VEC2_XLSR53':
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        feature = 1024
    elif wav2vec_name == 'WAV2VEC2_LARGE':
        bundle = torchaudio.pipelines.WAV2VEC2_LARGE
        feature = 1024
    elif wav2vec_name == 'WAV2VEC2_LARGE_LV60K':
        bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
        feature = 1024
    else:
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        feature = 768

    annotation_file = os.path.join(data_dir, 'labels', task + '_info.csv')
    wav_path = os.path.join(data_dir, 'audio', 'wav')
    h5p_path = '../competitions/A-VB2022/end-to-end_based/' + task + '/data_wav2vec'

    if task == 'type':
        metric  = UAR
        criteria = nn.CrossEntropyLoss()
        loss = 'ce'
        num_output = 8
        trainset = AVBWavType(annotation_file, wav_path, 'Train')
        validset = AVBWavType(annotation_file, wav_path, 'Val')
    else:
        if net_name == 'AvbWav2vecFeatureLstm':
            trainset = AVBH5py(annotation_file, h5p_path, 'Train')
            validset = AVBH5py(annotation_file, h5p_path, 'Val')
        else:
            trainset = AVBWav(annotation_file, wav_path, 'Train')
            validset = AVBWav(annotation_file, wav_path, 'Val')
        if loss == 'mse':
            criteria = nn.MSELoss()
        else:
            criteria = CCCLoss()
        metric  = AvgCCC

        if task == 'high':
            num_output = 10
        elif task == 'two':
            num_output = 2
        elif task == 'culture':
            num_output = 40

    trainldr = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=pad_collate)
    validldr = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate)

    start_epoch = 0

    if net_name == 'AvbWav2vecFeatureLstm':
        net = AvbWav2vecFeatureLstm(num_output)
    elif net_name == 'AvbWav2vec':
        net = AvbWav2vec(bundle, feature, num_output, freeze_extractor=True, layer=num_layers, loss=loss)
    else:
        net = AvbWav2vecLstm(bundle, feature, num_output, freeze_extractor=True, layer=num_layers, loss=loss)
    if resume != '':
        print("Resume form | {} ]".format(resume))
        net = load_state_dict(net, resume)

    net = nn.DataParallel(net).cuda()

    if args.sam:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(net.parameters(), base_optimizer, lr=learning_rate, momentum=0.9, weight_decay=1.0/batch_size)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=learning_rate, weight_decay=1.0/batch_size)

    best_performance = 0.0
    epoch_from_last_improvement = 0

    df = {}
    df['epoch'] = []
    df['lr'] = []
    df['train_loss'] = []
    df['val_loss'] = []
    df['val_metrics'] = []

    for epoch in range(start_epoch, epochs):
        lr = optimizer.param_groups[0]['lr']
        train_loss = train(net, trainldr, optimizer, epoch, epochs, learning_rate, criteria, task)
        val_loss, val_metrics = val(net, validldr, criteria, metric, task)

        infostr = {'Task {}: {},{:.5f},{:.5f},{:.5f},{:.5f}'
                .format(task,
                        epoch,
                        lr,
                        train_loss,
                        val_loss,
                        val_metrics)}
        print(infostr)

        os.makedirs(os.path.join('results', output_dir), exist_ok = True)

        if val_metrics >= best_performance:
            checkpoint = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join('results', output_dir, 'best_val_perform.pth'))
            best_performance = val_metrics
            epoch_from_last_improvement = 0
        else:
            epoch_from_last_improvement += 1

        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join('results', output_dir, 'cur_model.pth'))

        df['epoch'].append(epoch)
        df['lr'].append(lr)
        df['train_loss'].append(train_loss)
        df['val_loss'].append(val_loss)
        df['val_metrics'].append(val_metrics)

    df = pd.DataFrame(df)
    csv_name = os.path.join('results', output_dir, 'train.csv')
    df.to_csv(csv_name)

if __name__=="__main__":
    main()
