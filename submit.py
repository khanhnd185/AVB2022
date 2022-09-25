import os
from dataset import AVBFeature, AVBWav, AVBH5py, AVBWavType
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from helpers import *
from model import *
import torchaudio

def test(net, validldr):
    net.eval()
    names = []
    all_yhat = None

    for _, (x, y, leng, name) in enumerate(tqdm(validldr)):
        with torch.no_grad():
            x = x.float()
            y = y.float()
            x = x.cuda()
            y = y.cuda()
            yhat = net(x, leng)

            if all_yhat == None:
                all_yhat = yhat.clone()
            else:
                all_yhat = torch.cat((all_yhat, yhat), 0)
            names.extend(name)
    all_yhat = all_yhat.cpu().numpy()
    return names, all_yhat

def submit(task, output, names, predictions):
    task_header = {"two": "File_ID,Split,Valence,Arousal",
                   "type": "File_ID,Split,Voc_Type",
                   "high": "File_ID,Split,Awe,Excitement,Amusement,Awkwardness,Fear,Horror,Distress,Triumph,Sadness,Surprise",
                   "culture": "File_ID,Split,China_Awe,China_Excitement,China_Amusement,China_Awkwardness,China_Fear,China_Horror,China_Distress,China_Triumph,China_Sadness,United States_Awe,United States_Excitement,United States_Amusement,United States_Awkwardness,United States_Fear,United States_Horror,United States_Distress,United States_Triumph,United States_Sadness,South Africa_Awe,South Africa_Excitement,South Africa_Amusement,South Africa_Awkwardness,South Africa_Fear,South Africa_Horror,South Africa_Distress,South Africa_Triumph,South Africa_Sadness,Venezuela_Awe,Venezuela_Excitement,Venezuela_Amusement,Venezuela_Awkwardness,Venezuela_Fear,Venezuela_Horror,Venezuela_Distress,Venezuela_Triumph,Venezuela_Sadness,China_Surprise,United States_Surprise,South Africa_Surprise,Venezuela_Surprise",
                   }
    task_output = {"two": 2,
                   "type": 1,
                   "high": 10,
                   "culture": 40,
                   }
    
    types = ["Gasp","Laugh","Cry","Scream","Grunt","Groan","Pant","Other"]


    if task == "type":
        predictions = np.argmax(predictions, axis=1)

    with open(output, 'w') as f:
        f.write("{}\n".format(task_header[task]))
        for i, name in enumerate(names):
            f.write("{},Test".format(name))
            if task == "type":
                f.write(",{}".format(types[predictions[i]]))
            else:
                for j in range(task_output[task]):
                    f.write(",{}".format(predictions[i,j]))
            f.write("\n")




def main():
    parser = argparse.ArgumentParser(description='Train task seperately')

    parser.add_argument('--net', '-n', default='AvbWav2vecLstm', help='Net name')
    parser.add_argument('--input', '-i', default='', help='Input file')
    parser.add_argument('--pool', '-p', default='last', help='Pool type')
    parser.add_argument('--task', '-t', default='two', help='Task')
    parser.add_argument('--loss', '-L', default='ccc', help='Loss function')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--layer', '-l', type=int, default=12, help='Number of encoder layers')
    parser.add_argument('--datadir', '-d', default='../../../Data/A-VB/', help='Data folder path')
    parser.add_argument('--wav', '-w', default='WAV2VEC2_BASE', help='Wav2vec version')
    parser.add_argument('--output', '-o', default='1', help='Submission')

    args = parser.parse_args()
    task = args.task
    loss = args.loss
    pool = args.pool
    resume = args.input
    wav2vec_name = args.wav
    net_name = args.net
    data_dir = args.datadir
    batch_size = args.batch
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
        loss = 'ce'
        num_output = 8
        validset = AVBWavType(annotation_file, wav_path, 'Test')
    else:
        if net_name == 'AvbWav2vecFeatureLstm':
            validset = AVBH5py(annotation_file, h5p_path, 'Test')
        else:
            validset = AVBWav(annotation_file, wav_path, 'Test')

        if task == 'high':
            num_output = 10
        elif task == 'two':
            num_output = 2
        elif task == 'culture':
            num_output = 40

    validldr = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate)


    if net_name == 'AvbWav2vecFeatureLstm':
        net = AvbWav2vecFeatureLstm(num_output)
    elif net_name == 'AvbWav2vec':
        net = AvbWav2vec(bundle, feature, num_output, freeze_extractor=True, layer=num_layers, loss=loss)
    elif net_name == 'AvbWav2vecLstmPool':
        net = AvbWav2vecLstmPool(bundle, feature, num_output, freeze_extractor=True, layer=num_layers, loss=loss, pool=pool)
    else:
        net = AvbWav2vecLstm(bundle, feature, num_output, freeze_extractor=True, layer=num_layers, loss=loss)
    if resume != '':
        print("Resume form | {} ]".format(resume))
        net = load_state_dict(net, resume)

    net = nn.DataParallel(net).cuda()
    names, predictions = test(net, validldr)

    output_name = task + '_SclabCNU_' + args.output + '.csv'
    submit(task, output_name, names, predictions)

if __name__=="__main__":
    main()
