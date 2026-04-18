# fusion_runner.py
import os
import numpy as np
from typing import Dict, Tuple
from comad_wrapper import run_comad_for_class, MethodOutput as OutC
from patchcore_wrapper import run_patchcore_for_class, MethodOutput as OutP
from utils_area import compute_imagewise_retrieval_metrics

import numpy as np
from utils_area import compute_imagewise_retrieval_metrics

EPS = 1e-8

def _z_from_val(scores: np.ndarray, mu: float, sd: float) -> np.ndarray:
    if sd <= 0:
        sd = EPS
    return (scores - mu) / sd

def _fit_mu_sd_from_val(method_val, prefer_good: bool = True):
    """
    method_val: MethodOutput (paths, scores, labels, types)
    prefer_good=True면 good만으로 (mu, sd) 추정. 없으면 전체로 fallback.
    """
    v = np.asarray(method_val.scores, dtype=float)

    if prefer_good:
        good_mask = None
        # types 우선 (COMAD 쪽은 types 제공)
        if getattr(method_val, "types", None):
            t = np.array(method_val.types, dtype=object)
            good_mask = (t == "good")
        # 없으면 labels로 (False가 정상)
        if good_mask is None and getattr(method_val, "labels", None) is not None:
            good_mask = (np.asarray(method_val.labels) == False)

        if good_mask is not None and good_mask.any():
            v = v[good_mask]

    mu = float(np.mean(v)) if v.size else 0.0
    sd = float(np.std(v))  if v.size else 1.0
    if sd <= 0: sd = EPS
    return mu, sd

def fuse_and_eval_sum_zval(comad_test, patch_test, comad_val=None, patch_val=None, prefer_good=True):
    """
    comad_test, patch_test: MethodOutput (TEST 결과)
    comad_val, patch_val:  MethodOutput (VAL 결과; 각 방법별로 개별 정규화 파라미터 추정)
    prefer_good=True면 VAL의 good만으로 (mu, sd) 추정, 없으면 전체 사용
    """
    # 1) 길이 맞춰 자르기 (batch=1, shuffle=False 가정)
    sa = np.asarray(comad_test.scores, dtype=float)
    sb = np.asarray(patch_test.scores, dtype=float)
    n  = min(len(sa), len(sb))
    if n == 0:
        raise RuntimeError("합산할 공통 길이가 0입니다.")
    sa, sb = sa[:n], sb[:n]

    labels = np.asarray(comad_test.labels[:n], dtype=bool)
    types  = comad_test.types[:n] if comad_test.types else None

    # 2) 각 방법별 (mu, sd) 추정 — VAL 우선, 없으면 TEST의 good으로 fallback
    if comad_val is not None:
        mu_a, sd_a = _fit_mu_sd_from_val(comad_val, prefer_good=prefer_good)
    else:
        # fallback: TEST good만으로 추정 (라벨 누출 최소화 위해 good만)
        mu_a, sd_a = _fit_mu_sd_from_val(comad_test, prefer_good=True)

    if patch_val is not None:
        mu_b, sd_b = _fit_mu_sd_from_val(patch_val, prefer_good=prefer_good)
    else:
        mu_b, sd_b = _fit_mu_sd_from_val(patch_test, prefer_good=True)

    # 3) TEST 점수 z-정규화 → 합산
    sa_n = _z_from_val(sa, mu_a, sd_a)
    sb_n = _z_from_val(sb, mu_b, sd_b)
    fused = sa_n + sb_n

    # 4) 논리/구조 서브셋 AUROC
    res = {}
    logical_auc = None
    struct_auc  = None
    if types is not None:
        t = np.array(types, dtype=object)

        idx_log = (t == "good") | (t == "logical_anomalies")
        if idx_log.any():
            logical_auc = float(
                compute_imagewise_retrieval_metrics(fused[idx_log], labels[idx_log])["auroc"]
            )
            res["logical_only"] = logical_auc

        idx_stru = (t == "good") | (t == "stru")
        if idx_stru.any():
            struct_auc = float(
                compute_imagewise_retrieval_metrics(fused[idx_stru], labels[idx_stru])["auroc"]
            )
            res["struct_only"] = struct_auc

    # 5) overall = (logical_only + struct_only) / 2 (둘 중 하나만 있으면 그 값, 둘 다 없으면 전체)
    if (logical_auc is not None) and (struct_auc is not None):
        res["overall"] = float((logical_auc + struct_auc) / 2.0)
    elif (logical_auc is not None) or (struct_auc is not None):
        res["overall"] = float(logical_auc if logical_auc is not None else struct_auc)
    else:
        res["overall"] = float(compute_imagewise_retrieval_metrics(fused, labels)["auroc"])

    # (선택) 정규화 파라미터도 같이 리턴해두면 디버깅에 좋아요
    res["_zparams"] = {"comad": (mu_a, sd_a), "patch": (mu_b, sd_b)}
    return res



def fuse_and_eval_sum(comad_out, patch_out):
    # 1) 같은 순서 가정(batch=1, shuffle=False) → 앞에서부터 최소 길이만 사용
    sa = np.asarray(comad_out.scores, dtype=float)
    sb = np.asarray(patch_out.scores, dtype=float)
    n  = min(len(sa), len(sb))
    if n == 0:
        raise RuntimeError("합산할 공통 길이가 0입니다.")

    sa, sb = sa[:n], sb[:n]
    labels = np.asarray(comad_out.labels[:n], dtype=bool)
    types  = comad_out.types[:n] if comad_out.types else None

    # 2) ★ 정규화 없이 단순 합산
    fused = sa + sb

    # 3) 논리/구조 서브셋 AUROC 계산
    res = {}
    logical_auc = None
    struct_auc  = None

    if types is not None:
        t = np.array(types, dtype=object)

        idx_log = (t == "good") | (t == "logical_anomalies")
        if idx_log.any():
            logical_auc = float(
                compute_imagewise_retrieval_metrics(fused[idx_log], labels[idx_log])["auroc"]
            )
            res["logical_only"] = logical_auc

        idx_stru = (t == "good") | (t == "stru")
        if idx_stru.any():
            struct_auc = float(
                compute_imagewise_retrieval_metrics(fused[idx_stru], labels[idx_stru])["auroc"]
            )
            res["struct_only"] = struct_auc

    # 4) overall = logical_only와 struct_only의 평균(둘 중 하나만 있으면 그 값 사용)
    if (logical_auc is not None) and (struct_auc is not None):
        res["overall"] = float((logical_auc + struct_auc) / 2.0)
    elif (logical_auc is not None) or (struct_auc is not None):
        res["overall"] = float(logical_auc if logical_auc is not None else struct_auc)
    else:
        # types가 없거나 서브셋 비면 전체로 fallback
        res["overall"] = float(compute_imagewise_retrieval_metrics(fused, labels)["auroc"])

    return res


if __name__ == "__main__":
    sourcepath = "."
    data_path  = "./mvtec_loco_anomaly_detection"
    classlist  = ['breakfast_box', 'juice_bottle', 'screw_bag', 'pushpins', 'splicing_connectors']

    logs = []
    for cls in classlist:
        # TEST
        comad_test  = run_comad_for_class(sourcepath, cls, split="test")   # MethodOutput
        patch_test  = run_patchcore_for_class(data_path, cls, imagesize=224, resize=256, nn_k=1, split="test")

        # VAL (가능하면 둘 다 제공)
        comad_val   = run_comad_for_class(sourcepath, cls, split="val")    # 없으면 None
        patch_val   = run_patchcore_for_class(data_path, cls, imagesize=224, resize=256, nn_k=1, split="val")

        m = fuse_and_eval_sum_zval(comad_test, patch_test, comad_val, patch_val, prefer_good=True)

        print(f"[{cls}] AUROC overall={m['overall']:.4f}"
              + (f", logical={m.get('logical_only', np.nan):.4f}" if 'logical_only' in m else "")
              + (f", structural={m.get('struct_only', np.nan):.4f}" if 'struct_only' in m else "")
              + f"   (z COMAD μ,σ={m['_zparams']['comad'][0]:.4f},{m['_zparams']['comad'][1]:.4f}; "
                f"PATCH μ,σ={m['_zparams']['patch'][0]:.4f},{m['_zparams']['patch'][1]:.4f})")
        logs.append(m)

    # 매크로 평균
    metrics = {"overall": [], "logical_only": [], "struct_only": []}
    for m in logs:
        for k in metrics:
            if k in m and not np.isnan(m[k]):
                metrics[k].append(m[k])
    def safe_mean(vals): return float(np.mean(vals)) if vals else float('nan')
    print("\n=== Macro averages over classes (z-score via VAL) ===")
    print(f"Overall    mean AUROC = {safe_mean(metrics['overall']):.4f}  (n={len(metrics['overall'])})")
    if metrics["logical_only"]:
        print(f"Logical    mean AUROC = {safe_mean(metrics['logical_only']):.4f}  (n={len(metrics['logical_only'])})")
    if metrics["struct_only"]:
        print(f"Structural mean AUROC = {safe_mean(metrics['struct_only']):.4f}  (n={len(metrics['struct_only'])})")


    # 평균 내고 싶으면 여기서 추가
