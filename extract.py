
import os
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm.notebook import tqdm

def main():
    DEVICE = 'cuda'
    IMG_SIZE = 224

    parser = argparse.ArgumentParser(description='Generate image feature')
    parser.add_argument('--extractor', '-e', default='enet_b0_8_best_vgaf.pt', help='Extractor name')
    parser.add_argument('--output', '-o', default='saw2.pickle', help='Output file name')
    parser.add_argument('--batch', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--datadir', '-d', default='../../../Data/ABAW4/', help='Data folder path')
    parser.add_argument('--augment', '-a', action='store_true', help='Augment data')
    args = parser.parse_args()
    data_dir = args.datadir
    batch_size = args.batch
    output_name = args.output
    extractor_name = args.extractor

    if args.augment:
        transform = transforms.Compose(
            [
                transforms.Resize([int(IMG_SIZE * 1.02), int(IMG_SIZE * 1.02)]),
                transforms.RandomCrop([IMG_SIZE, IMG_SIZE]),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ]
        )

    feature_extractor_model = torch.load('./model/' + extractor_name)
    classifier_bias = feature_extractor_model.classifier[0].bias.cpu().data.numpy()
    classifier_weights = feature_extractor_model.classifier[0].weight.cpu().data.numpy()

    feature_extractor_model.classifier = torch.nn.Identity()
    feature_extractor_model = feature_extractor_model.to(DEVICE)
    feature_extractor_model.eval()
    image_dir = os.path.join(data_dir, 'cropped_aligned')

    imgs = []
    img_names = []
    X_global_features = []

    for filename in tqdm(os.listdir(image_dir)):
        frames_dir = os.path.join(image_dir, filename)    
        for img_name in os.listdir(frames_dir):
            if img_name.lower().endswith('.jpg'):
                img = Image.open(os.path.join(frames_dir, img_name))
                img_tensor = transform(img)
                if img.size:
                    img_names.append(filename + '/' + img_name)
                    imgs.append(img_tensor)
                    if len(imgs) >= batch_size:
                        features = feature_extractor_model(torch.stack(imgs, dim=0).to(DEVICE))
                        features = features.data.cpu().numpy()
                        
                        if len(X_global_features) == 0:
                            X_global_features = features
                        else:
                            X_global_features = np.concatenate((X_global_features, features), axis=0)
                        imgs = []

    if len(imgs) > 0:        
        features = feature_extractor_model(torch.stack(imgs, dim=0).to(DEVICE))
        features = features.data.cpu().numpy()

        if len(X_global_features) == 0:
            X_global_features = features
        else:
            X_global_features = np.concatenate((X_global_features, features), axis=0)

        imgs = []

    X_scores = np.dot(X_global_features, np.transpose(classifier_weights)) + classifier_bias
    feature_dict = {img_name: (global_features,scores) for img_name, global_features, scores in zip(img_names, X_global_features, X_scores)}

    with open(os.path.join(data_dir, output_name), 'wb') as handle:
        pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    main()
