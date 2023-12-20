import argparse

import torch
from utils import get_model, get_data, find_desired_samples, save_outputs, generate_representations, find_class_means
from kmeans_pytorch import kmeans


def get_evaluate_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--target-size', default=224, type=int)
    parser.add_argument('--data-path', default='Y:/AI team/datasets/artificial data/picture of a patio in 8k', type=str)

    parser.add_argument('--model-path', default='../EsViT_checkpoint_best.pth', type=str)
    parser.add_argument('--model-name', default='EsViT', type=str, choices=['EsViT', 'MAE'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--mixed-precision', default=True, type=bool)

    parser.add_argument('--quantile', default=0.9, type=float)
    parser.add_argument('--mode', default='unsupervised', type=str, choices=['unsupervised', 'supervised'])
    parser.add_argument('--use-classes-as-centroids', default=False, type=bool)

    parser.add_argument('--corpus-path', default='Y:/AI team/datasets/artificial data/picture of a patio in 8k',
                        type=str)
    parser.add_argument('--difficulty', default='hard', type=str, choices=['easy', 'hard'])
    parser.add_argument('--num-clusters', default=10, type=str)

    parser.add_argument('--draw-histogram', default=True, type=bool)

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_evaluate_parser()

    if args.mode == 'supervised':
        args.use_classes_as_centroids = True

    # initializing dataloader and model
    model = get_model(args)
    model.to(args.device)

    # generating representations of images
    dataloader, dataset = get_data(args, args.data_path)
    reps, indices, labels = generate_representations(args,
                                                     model=model,
                                                     dataloader=dataloader,
                                                     dataset=dataset,
                                                     desc='Generating Representations of Valid data')

    corpus_dataloader, corpus_dataset = get_data(args, args.corpus_path)
    corpus_reps, corpus_indices, corpus_labels = generate_representations(args, model=model,
                                                                          dataloader=corpus_dataloader,
                                                                          dataset=corpus_dataset,
                                                                          desc='Generating Representations of Corpus data')

    # clustering
    data_size, dims = reps.shape
    if args.num_clusters == 0:
        num_clusters = len(dataset.classes)
    else:
        num_clusters = args.num_clusters

    if args.use_classes_as_centroids:
        cluster_ids_x, cluster_centers = find_class_means(X=reps,
                                                          labels=labels,
                                                          num_clusters=num_clusters)
    else:
        cluster_ids_x, cluster_centers = kmeans(X=reps,
                                                num_clusters=num_clusters,
                                                distance='euclidean',
                                                device=torch.device(args.device),
                                                tol=1e-5)

    # finding and saving furtheset samples from centroid
    farthest_samples_paths = find_desired_samples(args=args,
                                                  reps=corpus_reps,
                                                  indices=corpus_indices,
                                                  labels=corpus_labels,
                                                  base_dataset=dataset,
                                                  target_dataset=corpus_dataset,
                                                  cluster_centers=cluster_centers,
                                                  cluster_ids_x=cluster_ids_x)

    save_outputs(dst_path=args.difficulty + ' samples',
                 dataset=corpus_dataset,
                 farthest_samples_paths=farthest_samples_paths)
