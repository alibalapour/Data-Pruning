import torch
from torchvision import datasets
from torch.utils.data import Dataset
from functools import partial
import torch.nn as nn
from torch.utils.data import DataLoader
from models.swin import SwinTransformer
from models.mae import get_mae_model
from torchvision import transforms
from torchvision import models as torchvision_models
import os
import shutil
from pathlib import Path
from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        # path = super(ReturnIndexDataset, self).samples[idx]
        return idx, img, lab, idx


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "deit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "deit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")


def find_class_means(X, labels, num_clusters):
    dim = X[0].shape[0]
    labels_sum = {i: torch.zeros(dim) for i in range(num_clusters)}
    labels_count = {i: 0 for i in range(num_clusters)}
    for i in range(len(X)):
        tensor = X[i]
        label = int(labels[i].item())
        labels_sum[label] += tensor
        labels_count[label] += 1
    labels_mean_tensor = torch.zeros((num_clusters, dim))
    for i in range(num_clusters):
        labels_mean_tensor[i] = labels_sum[i] / labels_count[i]
    return labels, labels_mean_tensor


def get_model(args):
    if args.model_name == 'EsViT':
        patch_size = 4
        model = SwinTransformer(
            img_size=args.target_size,
            in_chans=3,
            num_classes=0,
            patch_size=patch_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0,
            attn_drop_rate=0,
            drop_path_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            ape=False,
            patch_norm=True,
            use_dense_prediction=False,
        )
        model.cuda()
        load_pretrained_weights(model, args.model_path, 'teacher', 'swin_tiny', patch_size)

    elif args.model_name == 'MAE':
        model = get_mae_model(model_path=args.model_path, pretrained=True)
    else:
        model = None
    return model


def get_data(args, data_path):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(args.target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = ReturnIndexDataset(data_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        shuffle=True
    )
    return dataloader, dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def find_desired_samples(args, reps, indices, labels, base_dataset, target_dataset, cluster_centers, cluster_ids_x):
    res_values = []
    res_indices = []
    res_class_labels = []
    res_cluster_labels = []

    batch_size = args.batch_size
    num_clusters = len(cluster_centers)
    reps_dataset = CustomDataset(reps.detach())
    reps_dataloader = DataLoader(reps_dataset, batch_size=batch_size, shuffle=False)

    indices = torch.squeeze(indices)
    labels = torch.squeeze(labels)
    cluster_ids_x = torch.squeeze(cluster_ids_x)
    cluster_centers = cluster_centers.to(args.device)

    if args.mode == 'supervised':
        i = 0
        for tensor in tqdm(reps_dataloader, desc='Calculating norms'):
            num_classes = len(base_dataset.classes)
            tensor = tensor.to(args.device)
            label = labels[batch_size * i: (i + 1) * batch_size].to(torch.int64)
            label_mask = torch.nn.functional.one_hot(label, num_classes=num_classes).bool()
            norm_tensor = torch.linalg.norm(tensor.unsqueeze(dim=1) - cluster_centers.unsqueeze(dim=0), dim=2).detach().cpu()
            class_distance = torch.masked_select(norm_tensor, label_mask)
            other_distance = torch.masked_select(norm_tensor, ~label_mask).reshape((batch_size, num_classes-1))
            norm_tensor, norm_tensor_indecies = torch.sort(other_distance, dim=1)
            if args.difficulty == 'hard':
                res_values += (class_distance - norm_tensor[:, 0] - norm_tensor[:, 1]).tolist()
            elif args.difficulty == 'easy':
                res_values += (-class_distance).tolist()
            res_indices += (indices[batch_size * i: (i + 1) * batch_size]).tolist()
            res_class_labels += (labels[batch_size * i: (i + 1) * batch_size]).tolist()
            i += 1

        # reordering samples and finding quantiles baesd on each class
        class_scores = {k: [res_values[i] for i in range(len(res_values)) if int(res_class_labels[i]) == k] for k in
                        range(len(cluster_centers))}

        save_histograms(args, class_scores)

        quantiles = {k: torch.quantile(torch.tensor(class_scores[k]), q=args.quantile) for k in
                     range(len(base_dataset.classes)) if len(class_scores[k]) != 0}
        score_dicts = {int(res_indices[i]): (res_values[i], int(res_class_labels[i])) for i
                       in range(len(res_values))}

        results_based_on_class = {i: [] for i in range(len(target_dataset.classes))}

        # finding images which are in the quantile period
        for k, v in tqdm(score_dicts.items(), desc='Finding images in quntile'):
            if v[0] > quantiles[v[1]].item():
                results_based_on_class[v[1]].append(k)

    else:  # unsupervised
        # calculate norm
        i = 0
        for tensor in tqdm(reps_dataloader, desc='Calculating norms'):
            tensor = tensor.to(args.device)
            norm_tensor = torch.linalg.norm(tensor.unsqueeze(dim=1) - cluster_centers.unsqueeze(dim=0), dim=2).detach()
            norm_tensor, norm_tensor_indecies = torch.sort(norm_tensor, dim=1)
            if args.difficulty == 'hard':
                res_values += (norm_tensor[:, 0] - norm_tensor[:, 1] - norm_tensor[:, 2]).tolist()
            elif args.difficulty == 'easy':
                res_values += (-norm_tensor[:, 0]).tolist()
            res_indices += (indices[batch_size * i: (i + 1) * batch_size]).tolist()
            res_class_labels += (labels[batch_size * i: (i + 1) * batch_size]).tolist()
            res_cluster_labels += norm_tensor_indecies[:, 0].tolist()
            i += 1

        # reordering samples and finding quantiles baesd on each class
        cluster_scores = {k: [res_values[i] for i in range(len(res_values)) if int(res_cluster_labels[i]) == k] for k in
                          range(len(cluster_centers))}

        # save representation's distribution histogram
        save_histograms(args, cluster_scores)

        quantiles = {k: torch.quantile(torch.tensor(cluster_scores[k]), q=args.quantile) for k in
                     range(num_clusters) if len(cluster_scores[k]) != 0}
        score_dicts = {int(res_indices[i]): (res_values[i], int(res_class_labels[i]), int(res_cluster_labels[i])) for i
                       in
                       range(len(res_values))}
        results_based_on_class = {i: [] for i in range(len(target_dataset.classes))}

        # finding images which are in the quantile period
        for k, v in tqdm(score_dicts.items(), desc='Finding images in quntile'):
            if v[0] > quantiles[v[2]].item():
                results_based_on_class[v[1]].append(k)

    # find path of desired samples
    img_paths = {}
    for idx, img, label, ind in tqdm(target_dataset, desc='Gathering paths of desired samples'):
        image_path = target_dataset.samples[idx][0]
        if ind in results_based_on_class[label]:
            try:
                img_paths[label].append(image_path)
            except KeyError:
                img_paths[label] = [image_path]

    return img_paths


def save_histograms(args, cluster_scores):
    if args.draw_histogram:
        histograms_path = 'histograms'
        if os.path.exists(histograms_path):
            shutil.rmtree(histograms_path)
        os.mkdir(histograms_path)

        for cls, scores in cluster_scores.items():
            sns.distplot(scores)
            plt.title('class :' + str(cls))
            plt.savefig(os.path.join(histograms_path, 'class ' + str(cls) + '.jpg'))
            plt.clf()


def reverse_normalization(images):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    un_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return un_normalize(images)


def save_outputs(dst_path, dataset, farthest_samples_paths):
    try:
        shutil.rmtree(dst_path)
    except FileNotFoundError:
        pass
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    for cls in dataset.classes:
        Path(os.path.join(dst_path, str(cls))).mkdir(parents=True, exist_ok=True)
    for cls, paths in farthest_samples_paths.items():
        for i, path in enumerate(paths):
            shutil.copy(path, os.path.join(dst_path, dataset.classes[cls]))


def generate_representations(args, model, dataloader, dataset, desc=''):
    model.eval()

    reps = torch.zeros((len(dataloader) * args.batch_size, 1024))
    indices = torch.zeros((len(dataloader) * args.batch_size, 1))
    labels = torch.zeros((len(dataloader) * args.batch_size, 1))
    i = 0
    for idx, tensor, label, index in tqdm(dataloader, desc=desc):
        tensor = tensor.cuda()
        with autocast(enabled=args.mixed_precision):
            feats = model(tensor)
        reps[i * args.batch_size: min((i + 1) * args.batch_size, len(dataset))] = feats.detach().cpu()
        labels[i * args.batch_size: min((i + 1) * args.batch_size, len(dataset))] = label[:, None]
        indices[i * args.batch_size: min((i + 1) * args.batch_size, len(dataset))] = index[:, None]
        i += 1
    return reps, indices, labels
