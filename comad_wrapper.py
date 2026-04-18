# comad_wrapper.py
import os, glob, numpy as np
from dataclasses import dataclass
from typing import List, Optional
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from utils_area import (
    compute_imagewise_retrieval_metrics,
    get_area_list_new,
    get_area_only_histo,
    train_select_binary_offsets,
    test_select_binary_offsets,
)
from filter_algorithm import filter_bg_noise


@dataclass
class MethodOutput:
    paths:  List[str]
    scores: List[float]            # 높을수록 이상
    labels: List[bool]             # True=이상, False=정상
    types:  Optional[List[str]] = None   # "good" | "logical_anomalies" | "stru"


def _safe_sort_key(path: str):
    base = os.path.basename(path)
    return int(base) if base.isdigit() else float('inf')


def _test_area_color_component(area_all, color_all, files, sub, k_offset,
                               train_mean, train_std, cmean, cstd, dbscan, nn_connection):
    area, color, _, _ = test_select_binary_offsets(
        files, sub, k_offset, train_mean, train_std, cmean, cstd
    )
    area_all.append(area)
    color_all.append(color)
    histo_np = get_area_only_histo(files, sub, k_offset, dbscan, nn_connection)
    return area_all, color_all, histo_np


def _test_global_info(score_all: np.ndarray,
                      area_all: List[np.ndarray],
                      color_all: List[np.ndarray],
                      nn_train_global: NearestNeighbors) -> np.ndarray:
    area_np  = np.concatenate(area_all, axis=1) if len(area_all)  else np.zeros((0,0))
    color_np = np.concatenate(color_all, axis=1) if len(color_all) else np.zeros((0,0))
    if area_np.size == 0 or color_np.size == 0:
        return score_all
    te_glob  = np.concatenate((area_np, color_np), axis=1)
    dis, _   = nn_train_global.kneighbors(te_glob)
    return score_all + np.mean(dis, axis=1)


def run_comad_for_class(
    sourcepath: str,
    classname: str,
    *,
    split: str = "test",   # "test" | "val"
    knn: int = 5,
    alpha: float = 0.5
) -> MethodOutput:
    """
    split="test":  good / logical_anomalies / stru 3개 스플릿 사용
    split="val":   validation 폴더를 good으로 간주해 점수를 계산 (라벨=전부 False, type="good")
    """

    # ----- 1) 파일 목록 -----
    base = os.path.join(sourcepath, f'{classname}_heat')

    train_dir = os.path.join(base, 'train')
    trainfiles  = sorted(glob.glob(os.path.join(train_dir, '*')), key=_safe_sort_key)
    assert len(trainfiles) > 0, f"[COMAD] trainfiles empty: {train_dir}"

    if split.lower() == "val":
        # VAL은 validation 폴더만 good 취급
        good_dir  = os.path.join(base, 'validation')
        tegoodfiles = sorted(glob.glob(os.path.join(good_dir, '*')), key=_safe_sort_key)
        telogfiles, testrufiles = [], []
        assert len(tegoodfiles) > 0, f"[COMAD] validation empty: {good_dir}"
    else:
        good_dir  = os.path.join(base, 'test', 'good')
        log_dir   = os.path.join(base, 'test', 'logical_anomalies')
        stru_dir  = os.path.join(base, 'test', 'stru')
        tegoodfiles = sorted(glob.glob(os.path.join(good_dir,  '*')), key=_safe_sort_key)
        telogfiles  = sorted(glob.glob(os.path.join(log_dir,   '*')), key=_safe_sort_key)
        testrufiles = sorted(glob.glob(os.path.join(stru_dir,  '*')), key=_safe_sort_key)
        assert len(tegoodfiles) > 0, f"[COMAD] tegoodfiles empty: {good_dir}"

    # ----- 2) 서브컴포넌트 -----
    subs = filter_bg_noise(sourcepath, classname)
    assert len(subs) > 0, f"[COMAD] filter_bg_noise returned empty for {classname}"

    # ----- 3) 누적 컨테이너 -----
    score_good = 0.0
    score_log  = 0.0
    score_stru = 0.0

    area_train_all, color_train_all = [], []
    area_good_all,  color_good_all  = [], []
    area_log_all,   color_log_all   = [], []
    area_stru_all,  color_stru_all  = [], []

    # ----- 4) 메인 루프 -----
    for sub in subs:
        area_train, train_mean, train_std, k_offset = train_select_binary_offsets(trainfiles, sub)
        area_train_all.append(area_train)

        comp_area_list = get_area_list_new(trainfiles, sub, k_offset)
        comp_area = np.asarray(comp_area_list)
        assert comp_area.size > 0, f"[COMAD] comp_area empty for sub={sub}"

        dbscan_r = float(np.mean(comp_area)) * 0.1
        dbscan   = DBSCAN(eps=dbscan_r, min_samples=10).fit(comp_area)
        nn_conn  = NearestNeighbors(n_neighbors=knn).fit(comp_area)

        train_histo_np = get_area_only_histo(trainfiles, sub, k_offset, dbscan, nn_conn)
        nn_conn_histo  = NearestNeighbors(n_neighbors=knn).fit(train_histo_np)

        # color train
        _, color_train, cmean, cstd = test_select_binary_offsets(
            trainfiles, sub, k_offset, train_mean, train_std, 0, 0
        )
        color_train_all.append(color_train)

        # 대상 split
        area_good_all,  color_good_all,  good_histo_np = _test_area_color_component(
            area_good_all,  color_good_all,  tegoodfiles,  sub, k_offset,
            train_mean, train_std, cmean, cstd, dbscan, nn_conn
        )

        if len(telogfiles) > 0:
            area_log_all,   color_log_all,   log_histo_np = _test_area_color_component(
                area_log_all, color_log_all, telogfiles, sub, k_offset,
                train_mean, train_std, cmean, cstd, dbscan, nn_conn
            )
        else:
            log_histo_np = np.zeros((0, train_histo_np.shape[1]), dtype=float)

        if len(testrufiles) > 0:
            area_stru_all,  color_stru_all,  stru_histo_np = _test_area_color_component(
                area_stru_all, color_stru_all, testrufiles, sub, k_offset,
                train_mean, train_std, cmean, cstd, dbscan, nn_conn
            )
        else:
            stru_histo_np = np.zeros((0, train_histo_np.shape[1]), dtype=float)

        # component distance 누적
        div = float(train_histo_np.shape[1] * train_histo_np.shape[1])
        if good_histo_np.size:
            score_good += np.mean(nn_conn_histo.kneighbors(good_histo_np)[0], axis=1) * alpha / div
        if log_histo_np.size:
            score_log  += np.mean(nn_conn_histo.kneighbors(log_histo_np)[0],  axis=1) * alpha / div
        if stru_histo_np.size:
            score_stru += np.mean(nn_conn_histo.kneighbors(stru_histo_np)[0], axis=1) * alpha / div

    # ----- 5) 글로벌 NN 보정 -----
    assert len(area_train_all) > 0 and len(color_train_all) > 0, "[COMAD] no train stacks"
    nn_train_global = NearestNeighbors(n_neighbors=knn).fit(
        np.concatenate(
            (np.concatenate(area_train_all, axis=1), np.concatenate(color_train_all, axis=1)),
            axis=1,
        )
    )

    score_good = _test_global_info(score_good, area_good_all, color_good_all, nn_train_global)
    score_log  = _test_global_info(score_log,  area_log_all,  color_log_all,  nn_train_global) if len(telogfiles)>0  else np.array([], dtype=float)
    score_stru = _test_global_info(score_stru, area_stru_all, color_stru_all, nn_train_global) if len(testrufiles)>0 else np.array([], dtype=float)

    # ----- 6) 출력 구성 -----
    all_paths  = tegoodfiles + telogfiles + testrufiles
    all_scores = np.concatenate([score_good, score_log, score_stru], axis=0).astype(float).tolist()
    if split.lower() == "val":
        # VAL은 전부 정상 취급
        all_labels = [False] * len(all_paths)
        all_types  = ["good"] * len(all_paths)
    else:
        all_labels = [False]*len(tegoodfiles) + [True]*len(telogfiles) + [True]*len(testrufiles)
        all_types  = (["good"]*len(tegoodfiles)
                     +["logical_anomalies"]*len(telogfiles)
                     +["stru"]*len(testrufiles))

    assert len(all_scores) == len(all_paths) == len(all_labels) == len(all_types), \
        f"[COMAD] length mismatch: scores={len(all_scores)}, paths={len(all_paths)}, labels={len(all_labels)}, types={len(all_types)}"

    return MethodOutput(paths=all_paths, scores=all_scores, labels=all_labels, types=all_types)
