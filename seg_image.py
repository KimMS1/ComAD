from modules import DinoFeaturizer
from dataset import MVTecLocoDataset
from torch.utils.data import DataLoader, Subset
import torch
from sampler import GreedyCoresetSampler
import torch.nn.functional as F
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from crf import dense_crf
from torchvision import transforms
import random
import glob

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
i_m = np.array(IMAGENET_MEAN)[:, None, None]
i_std = np.array(IMAGENET_STD)[:, None, None]

def build_validation_loader(dataset_root, image_size, val_ratio=0.2, seed=0):
    """
    1) dataset_root/validation/good 이 있으면 그걸 사용
    2) dataset_root/validation 루트에 이미지가 직접 있으면 그걸 사용
    3) 둘 다 없으면 test/good에서 val_ratio 만큼 샘플링해 Subset으로 사용
    """
    val_good_dir = os.path.join(dataset_root, "validation", "good")
    val_root_dir = os.path.join(dataset_root, "validation")

    # case 1: validation/good 구조 (권장)
    if os.path.isdir(val_good_dir) and len(os.listdir(val_good_dir)) > 0:
        ds = MVTecLocoDataset(root_dir=dataset_root, category='validation/good', resize_shape=image_size)
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0), "validation(from validation/good)"

    # case 2: validation 루트에 이미지가 직접 있는 구조
    if os.path.isdir(val_root_dir):
        # 이미지 파일이 실제로 있는지 확인
        exts = ("*.jpg","*.png","*.jpeg","*.bmp","*.tif","*.tiff")
        files = []
        for e in exts:
            files += glob.glob(os.path.join(val_root_dir, e))
        if len(files) > 0:
            # dataset이 'validation' 카테고리를 지원한다면 이 분기 허용
            ds = MVTecLocoDataset(root_dir=dataset_root, category='validation', resize_shape=image_size)
            return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0), "validation(from validation root)"

    # fallback: test/good → subset
    ds_test_good = MVTecLocoDataset(root_dir=dataset_root, category='test/good', resize_shape=image_size)
    n = len(ds_test_good)
    if n == 0:
        return None, "no_validation_and_no_test_good"

    k = max(1, int(round(n * val_ratio)))
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    sub_idx = idx[:k]
    subset_val = Subset(ds_test_good, sub_idx)
    return DataLoader(subset_val, batch_size=1, shuffle=False, num_workers=0), f"validation(from test/good subset: {k}/{n})"

def run() -> None:
    # --- Train / Test Datasets & Loaders
    dataset_train = MVTecLocoDataset(root_dir=dataset_root, category='train/good', resize_shape=image_size)
    dataloader = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)

    dataset_test_logical = MVTecLocoDataset(root_dir=dataset_root, category='test/logical_anomalies', resize_shape=image_size)
    dataloader_test_logical = DataLoader(dataset_test_logical, batch_size=1, shuffle=False, num_workers=0)

    dataset_test_good = MVTecLocoDataset(root_dir=dataset_root, category='test/good', resize_shape=image_size)
    dataloader_test_good = DataLoader(dataset_test_good, batch_size=1, shuffle=False, num_workers=0)

    dataset_test_stru = MVTecLocoDataset(root_dir=dataset_root, category='test/structural_anomalies', resize_shape=image_size)
    dataloader_test_stru = DataLoader(dataset_test_stru, batch_size=1, shuffle=False, num_workers=0)

    # --- Validation Loader (dataset/validation 우선, 없으면 test/good 서브셋)
    dataloader_val, val_source = build_validation_loader(dataset_root, image_size, val_ratio=val_ratio, seed=val_seed)
    print(f"[{classname}] validation source: {val_source}")

    # --- Feature Extractor
    net = DinoFeaturizer().cuda()

    # --- Train feature sampling
    train_feature_list = []
    greedsampler_perimg = GreedyCoresetSampler(percentage=0.01, device='cuda')
    if StartTrain:
        for i, Img in enumerate(dataloader):
            with torch.no_grad():
                image = Img['image'].cuda()
                feats0, f_lowdim = net(image)
                feats = feats0.squeeze().reshape(feats0.shape[1], -1).permute(1, 0)
                feats_sample = greedsampler_perimg.run(feats)
                train_feature_list.append(feats_sample)

        train_features = torch.cat(train_feature_list, dim=0)
        train_features = F.normalize(train_features, dim=1)
        torch.save(train_features.cpu(), f'{classname}.pth')

        train_features_np = train_features.cpu().numpy()
        kmeans = KMeans(init='k-means++', n_clusters=num_cluster)
        c = kmeans.fit(train_features_np)
        cluster_centers = torch.from_numpy(c.cluster_centers_)
        torch.save(cluster_centers.cpu(), f'{classname}_k{num_cluster}.pth')
        train_features_sampled = cluster_centers.cuda().unsqueeze(0).unsqueeze(0).permute(0, 3, 1, 2)
    else:
        train_features = torch.load(f'{classname}.pth').cuda()
        train_features_np = train_features.cpu().numpy()
        kmeans = KMeans(init='k-means++', n_clusters=num_cluster)
        c = kmeans.fit(train_features_np)
        cluster_centers = torch.from_numpy(c.cluster_centers_)
        train_features_sampled = cluster_centers.cuda().unsqueeze(0).unsqueeze(0).permute(0, 3, 1, 2)

    # --- Save heat/seg results
    savepath = f'{classname}_heat'
    os.makedirs(savepath, exist_ok=True)

    train_savepath = f'{savepath}/train'
    os.makedirs(train_savepath, exist_ok=True)
    save_img(dataloader, train_features_sampled, net, train_savepath)

    # validation 저장 (핵심 추가)
    if dataloader_val is not None:
        val_savepath = f'{savepath}/validation'
        os.makedirs(val_savepath, exist_ok=True)
        save_img(dataloader_val, train_features_sampled, net, val_savepath)

    test_logical_savepath = f'{savepath}/test/logical_anomalies'
    os.makedirs(test_logical_savepath, exist_ok=True)
    save_img(dataloader_test_logical, train_features_sampled, net, test_logical_savepath)

    test_good_savepath = f'{savepath}/test/good'
    os.makedirs(test_good_savepath, exist_ok=True)
    save_img(dataloader_test_good, train_features_sampled, net, test_good_savepath)

    test_stru_savepath = f'{savepath}/test/stru'
    os.makedirs(test_stru_savepath, exist_ok=True)
    save_img(dataloader_test_stru, train_features_sampled, net, test_stru_savepath)

def save_img(dataloader, train_features_sampled, net, savapath):
    for i, Img in enumerate(dataloader):
        image = Img['image']          # tensor [1,3,H,W]
        # 원본 저장용 (ToPILImage)
        imageo = Img['image1'][0, :, :, :]
        imageo = unloader(imageo)

        heatmap, heatmap_intra = get_heatmaps(image, train_features_sampled, net)

        img_savepath = f'{savapath}/{i}'
        os.makedirs(img_savepath, exist_ok=True)

        # 원본 저장
        imageo.save(f'{img_savepath}/imgo.jpg')  # 참고용
        see_image(image, heatmap, img_savepath, heatmap_intra)  # img.jpg, heatresult*.jpg 들 생성

def get_heatmaps(img, query_feature, net):
    with torch.no_grad():
        feats1, f1_lowdim = net(img.cuda())
    sfeats1 = query_feature
    attn_intra = torch.einsum("nchw,ncij->nhwij",
                              F.normalize(sfeats1, dim=1),
                              F.normalize(feats1, dim=1))
    attn_intra -= attn_intra.mean([3, 4], keepdims=True)
    attn_intra = attn_intra.clamp(0).squeeze(0)
    heatmap_intra = F.interpolate(attn_intra, img.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()

    img_crf = img.squeeze()
    crf_result = dense_crf(img_crf, heatmap_intra)
    heatmap_intra = torch.from_numpy(crf_result)

    d = heatmap_intra.argmax(dim=0)
    d = d[None, None, :, :].repeat(1, 3, 1, 1)

    seg_map = torch.zeros([1, 3, d.shape[2], d.shape[3]], dtype=torch.int64)
    for color in range(query_feature.shape[3]):
        seg_map = torch.where(d == color, color_tensor[color], seg_map)
    return seg_map, heatmap_intra

def see_image(data, heatmap, savepath, heatmap_intra):
    # 원본 저장
    data = data[0].cpu().numpy()
    data = np.clip((data * i_std + i_m) * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{savepath}/img.jpg', data)

    # seg_map 저장 (RGB 클래스 인덱스 컬러)
    heatmap_np = heatmap[0].cpu().numpy().transpose(1, 2, 0)
    cv2.imwrite(f'{savepath}/heatresult.jpg', heatmap_np)

    # 각 클러스터 채널별 히트 저장 (COMAD에서 쓰는 heatresult{sub}.jpg)
    for i in range(heatmap_intra.shape[0]):
        heat = heatmap_intra[i, :, :].cpu().numpy()
        heat = np.round(heat * 128).astype(np.uint8)
        cv2.imwrite(f'{savepath}/heatresult{i}.jpg', heat)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str, default='./mvtec_loco_anomaly_detection/')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation ratio if dataset/validation missing')
    parser.add_argument('--val_seed', type=int, default=0, help='random seed for validation subset')
    args = parser.parse_args()

    unloader = transforms.ToPILImage()
    StartTrain = True
    image_size = 224
    color_list = [[127, 123, 229], [195, 240, 251], [146, 223, 255], [243, 241, 230], [224, 190, 144], [178, 116, 75]]
    color_tensor = torch.tensor(color_list)[:, :, None, None].repeat(1, 1, image_size, image_size)
    num_cluster = 5

    # 외부에서 받는 val 설정
    val_ratio = args.val_ratio
    val_seed = args.val_seed

    classlist = ['screw_bag','breakfast_box', 'juice_bottle', 'pushpins', 'splicing_connectors']
    for classname in classlist:
        dataset_root = f'{args.datasetpath}/{classname}/'
        run()
