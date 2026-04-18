# patchcore_wrapper.py
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

# patchcore 내부 모듈
import patchcoremain.src.patchcore.backbones as backbones
import patchcoremain.src.patchcore.common as common
import patchcoremain.src.patchcore.patchcore as patchcore
import patchcoremain.src.patchcore.sampler as sampler_lib
import patchcoremain.src.patchcore.utils as U

# MVTec LOCO Dataset
from patchcoremain.src.patchcore.datasets.mvtec import MVTecDataset, DatasetSplit


@dataclass
class MethodOutput:
    paths:  List[str]
    scores: List[float]              # 높을수록 이상
    labels: List[bool]               # True=이상, False=정상
    types:  Optional[List[str]] = None  # "good" | "logical_anomalies" | "stru"


def _to_types(label_strs: List[str]) -> List[str]:
    # MVTec LOCO dataset은 "good" / "logical_anomalies" / "structural_anomalies" 등을 사용
    out = []
    for s in label_strs:
        if s == "structural_anomalies":
            out.append("stru")
        else:
            out.append(s)  # "good" 또는 "logical_anomalies"
    return out


def run_patchcore_for_class(
    data_path: str,
    classname: str,
    *,
    split: str = "test",                 # "test" | "val"
    backbone_name="wideresnet50",
    layers=("layer2", "layer3"),
    imagesize=224,
    resize=256,
    batch_size=2,
    num_workers=8,
    sampler_name="approx_greedy_coreset",
    sampler_pct=0.1,
    nn_k=1,
    device_id=0,
    seed=0,
) -> MethodOutput:

    device = U.set_torch_device([device_id])
    U.fix_seeds(seed, device)

    # ----- 데이터셋 & 로더 -----
    train_dataset = MVTecDataset(
        data_path, classname=classname, resize=resize, imagesize=imagesize,
        split=DatasetSplit.TRAIN, seed=seed, augment=False
    )

    split_map = {"test": DatasetSplit.TEST, "val": DatasetSplit.VAL}
    if split.lower() not in split_map:
        raise ValueError(f"[PatchCore] split must be 'test' or 'val', got {split}")
    eval_split = split_map[split.lower()]

    if split.lower() == "val":
        # ✅ 강제 경로 우회: {classname}/validation/good 만 평가 셋으로 사용
        val_root = os.path.join(data_path, classname, "validation", "good")
        if not os.path.isdir(val_root):
            raise AssertionError(f"[PatchCore] validation dir not found: {val_root}")

        # MVTecDataset이 VAL을 제대로 지원하지 않는 경우를 우회하기 위해
        # TRAIN과 동일한 변환/전처리를 사용하되, data_to_iterate를 우리가 강제로 구성
        eval_dataset = MVTecDataset(
            data_path, classname=classname, resize=resize, imagesize=imagesize,
            split=DatasetSplit.TRAIN, seed=seed, augment=False
        )
        # 이미지 리스트를 직접 바꿔치기 (label은 모두 'good', mask는 None)
        img_paths = []
        for fname in sorted(os.listdir(val_root)):
            # 확장자 필터 필요 시 추가
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                img_paths.append(os.path.join(val_root, fname))
        if len(img_paths) == 0:
            raise AssertionError(f"[PatchCore] no images in validation dir: {val_root}")

        # data_to_iterate: (classname, label_str, image_path, mask_path)
        eval_dataset.data_to_iterate = [
            (classname, "good", p, None) for p in img_paths
        ]
    else:
        # 기존 TEST 경로는 그대로
        eval_dataset = MVTecDataset(
            data_path, classname=classname, resize=resize, imagesize=imagesize,
            split=eval_split, seed=seed
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # ----- 샘플러 -----
    if sampler_name == "identity":
        feat_sampler = sampler_lib.IdentitySampler()
    elif sampler_name == "greedy_coreset":
        feat_sampler = sampler_lib.GreedyCoresetSampler(sampler_pct, device)
    else:
        feat_sampler = sampler_lib.ApproximateGreedyCoresetSampler(sampler_pct, device)

    # ----- 백본 & PatchCore -----
    bb = backbones.load(backbone_name)
    faiss_nn = common.FaissNN(False, 8)

    P = patchcore.PatchCore(device)
    P.load(
        backbone=bb,
        layers_to_extract_from=list(layers),
        device=device,
        input_shape=eval_dataset.imagesize,
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        featuresampler=feat_sampler,
        anomaly_scorer_num_nn=nn_k,
        nn_method=faiss_nn,
    )

    # ----- 학습 & 예측 -----
    with (torch.cuda.device(device.index) if "cuda" in device.type else U.contextlib.suppress()):
        torch.cuda.empty_cache()
        P.fit(train_loader)
        torch.cuda.empty_cache()
        scores, segmentations, labels_gt, masks_gt = P.predict(eval_loader)

    # ----- 경로/라벨/타입 -----
    # dataset.data_to_iterate 항목: (classname, label_str, image_path, mask_path)
    label_strs  = [x[1] for x in eval_loader.dataset.data_to_iterate]
    image_paths = [x[2] for x in eval_loader.dataset.data_to_iterate]
    types = _to_types(label_strs)
    anomaly_labels = [s != "good" for s in types]

    # numpy → list 보장
    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    return MethodOutput(
        paths=image_paths,
        scores=scores,
        labels=anomaly_labels,
        types=types
    )
