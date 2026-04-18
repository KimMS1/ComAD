import glob
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from utils_area import compute_imagewise_retrieval_metrics, get_area_list_new, get_area_only_histo, train_select_binary_offsets, test_select_binary_offsets
from filter_algorithm import filter_bg_noise


def test_area_color_component(area_tegood_all, color_tegood_all, files, sub, k_offset, train_mean, train_std, cmean,
                              cstd, dbscan, nn_connection):
    area_tegood, color_tegood, _, _ = test_select_binary_offsets(files, sub, k_offset, train_mean, train_std, cmean,
                                                                 cstd)
    area_tegood_all.append(area_tegood)
    color_tegood_all.append(color_tegood)
    tegood_histo_numpy = get_area_only_histo(files, sub, k_offset, dbscan, nn_connection)
    return area_tegood_all, color_tegood_all, tegood_histo_numpy


def test_global_info(score_all, area_tegood_all, color_tegood_all, nn_train_global):
    if len(area_tegood_all) == 0 or len(color_tegood_all) == 0:
        return score_all
    area_tegood_all_numpy = np.concatenate(area_tegood_all, axis=1)
    color_tegood_all_numpy = np.concatenate(color_tegood_all, axis=1)
    tegood_global = np.concatenate((area_tegood_all_numpy, color_tegood_all_numpy), axis=1)
    dis_tegood_global, _ = nn_train_global.kneighbors(tegood_global)
    dis_tegood_global = np.mean(dis_tegood_global, axis=1)
    score_tegood_all = score_all + dis_tegood_global
    return score_tegood_all


def safe_sort_key(path):
    """Extract numeric basename safely for sorting"""
    base = os.path.basename(path)
    return int(base) if base.isdigit() else float('inf')


# ---------------------- NEW: validation 폴더 없을 때 good에서 샘플링 ----------------------
def select_split_files(sourcepath, classname, val_ratio=0.2, seed=0):
    """
    반환:
        trainfiles, tegoodfiles, telogicalfiles, testrufiles, valfiles
    규칙:
        - validation 폴더가 있으면 valfiles 그대로 사용
        - 없거나 비어 있으면 good 중 일부를 랜덤 샘플링하여 valfiles로 사용하고, 나머지를 tegood로 사용
    """
    rng = np.random.RandomState(seed)

    train_dir = os.path.join(sourcepath, f'{classname}_heat', 'train')
    good_dir  = os.path.join(sourcepath, f'{classname}_heat', 'test', 'good')
    log_dir   = os.path.join(sourcepath, f'{classname}_heat', 'test', 'logical_anomalies')
    stru_dir  = os.path.join(sourcepath, f'{classname}_heat', 'test', 'stru')
    val_dir   = os.path.join(sourcepath, f'{classname}_heat', 'test', 'validation')

    trainfiles  = sorted(glob.glob(os.path.join(train_dir, '*')), key=safe_sort_key)
    tegoodfiles = sorted(glob.glob(os.path.join(good_dir,  '*')), key=safe_sort_key)
    telogfiles  = sorted(glob.glob(os.path.join(log_dir,   '*')), key=safe_sort_key)
    testrufiles = sorted(glob.glob(os.path.join(stru_dir,  '*')), key=safe_sort_key)
    valfiles    = sorted(glob.glob(os.path.join(val_dir,   '*')), key=safe_sort_key)

    # validation 폴더가 없거나 비어 있으면 good에서 샘플링
    if len(valfiles) == 0:
        if len(tegoodfiles) == 0:
            raise RuntimeError(f"[COMAD] No good files to sample pseudo-validation for {classname}.")
        n_good = len(tegoodfiles)
        n_val  = max(1, int(round(n_good * val_ratio)))  # 최소 1장은 VAL로
        idx = np.arange(n_good)
        rng.shuffle(idx)
        val_idx = set(idx[:n_val].tolist())
        new_val = [tegoodfiles[i] for i in range(n_good) if i in val_idx]
        new_good = [tegoodfiles[i] for i in range(n_good) if i not in val_idx]
        valfiles = new_val
        tegoodfiles = new_good
        print(f"[COMAD:{classname}] validation folder missing → sampled {len(valfiles)}/{n_good} from good as VAL.")
    else:
        print(f"[COMAD:{classname}] using real validation folder: {val_dir} ({len(valfiles)} files)")

    if len(trainfiles) == 0:
        raise RuntimeError(f"[COMAD] trainfiles empty: {train_dir}")
    if len(tegoodfiles) == 0:
        raise RuntimeError(f"[COMAD] tegoodfiles empty (after val split): {good_dir}")

    return trainfiles, tegoodfiles, telogfiles, testrufiles, valfiles
# ------------------------------------------------------------------------------------------


subdict = {}
class_num = 0
auroc_log = 0
auroc_stru = 0
classlist = ['breakfast_box', 'juice_bottle', 'screw_bag', 'pushpins', 'splicing_connectors']
sourcepath = '.'

for classname in classlist:
    # --- 여기만 바뀜: 파일 스플릿을 유틸로 통일 (VAL fallback 포함)
    trainfiles, tegoodfiles, telogicalfiles, testrufiles, valfiles = select_split_files(
        sourcepath, classname, val_ratio=0.2, seed=0
    )

    knn = 5
    score_tegood_all = 0
    score_telogical_all = 0
    score_testru_all = 0
    score_val_all = 0

    area_train_all = []
    area_tegood_all = []
    area_tlogical_all = []
    area_stru_all = []
    area_val_all = []

    color_train_all = []
    color_tegood_all = []
    color_tlogical_all = []
    color_stru_all = []
    color_val_all = []

    # component
    subdict[f'{classname}'] = filter_bg_noise(sourcepath, classname)

    for sub in subdict[f'{classname}']:
        # train area
        area_train, train_mean, train_std, k_offset = train_select_binary_offsets(trainfiles, sub)
        nn_area = NearestNeighbors(n_neighbors=knn)
        nn_area.fit(area_train)
        area_train_all.append(area_train)

        # train component
        component_area_list = get_area_list_new(trainfiles, sub, k_offset)
        component_area = np.asarray(component_area_list)
        component_area_mean = np.mean(component_area)
        dbscan_r = component_area_mean * 0.1
        dbscan_min = 10
        dbscan = DBSCAN(eps=dbscan_r, min_samples=dbscan_min)

        dbscan.fit(component_area)
        nn_connection = NearestNeighbors(n_neighbors=knn)
        nn_connection.fit(component_area)

        train_histo_numpy = get_area_only_histo(trainfiles, sub, k_offset, dbscan, nn_connection)
        nn_connection_histo = NearestNeighbors(n_neighbors=knn)
        nn_connection_histo.fit(train_histo_numpy)

        # train color
        _, color_train, cmean, cstd = test_select_binary_offsets(
            trainfiles, sub, k_offset, train_mean, train_std, 0, 0
        )
        color_train_all.append(color_train)

        # test_good
        area_tegood_all, color_tegood_all, tegood_histo_numpy = test_area_color_component(
            area_tegood_all, color_tegood_all, tegoodfiles, sub, k_offset, train_mean, train_std, cmean, cstd, dbscan, nn_connection
        )
        # test_logical
        area_tlogical_all, color_tlogical_all, telogical_histo_numpy = test_area_color_component(
            area_tlogical_all, color_tlogical_all, telogicalfiles, sub, k_offset, train_mean, train_std, cmean, cstd, dbscan, nn_connection
        )
        # test_stru
        area_stru_all, color_stru_all, tstru_histo_numpy = test_area_color_component(
            area_stru_all, color_stru_all, testrufiles, sub, k_offset, train_mean, train_std, cmean, cstd, dbscan, nn_connection
        )
        # NEW: validation (good으로 취급)
        area_val_all, color_val_all, val_histo_numpy = test_area_color_component(
            area_val_all, color_val_all, valfiles, sub, k_offset, train_mean, train_std, cmean, cstd, dbscan, nn_connection
        )

        # component distances
        alpha = 0.5
        denom = (train_histo_numpy.shape[1] * train_histo_numpy.shape[1])

        # good
        if tegood_histo_numpy.size:
            dis_tegood, _ = nn_connection_histo.kneighbors(tegood_histo_numpy)
            dis_tegood = np.mean(dis_tegood, axis=1) * alpha / denom
            score_tegood_all = score_tegood_all + dis_tegood

        # logical
        if telogical_histo_numpy.size:
            dis_telogical, _ = nn_connection_histo.kneighbors(telogical_histo_numpy)
            dis_telogical = np.mean(dis_telogical, axis=1) * alpha / denom
            score_telogical_all = score_telogical_all + dis_telogical

        # stru
        if tstru_histo_numpy.size:
            dis_testru, _ = nn_connection_histo.kneighbors(tstru_histo_numpy)
            dis_testru = np.mean(dis_testru, axis=1) * alpha / denom
            score_testru_all = score_testru_all + dis_testru

        # NEW: val (전부 good 취급)
        if val_histo_numpy.size:
            dis_val, _ = nn_connection_histo.kneighbors(val_histo_numpy)
            dis_val = np.mean(dis_val, axis=1) * alpha / denom
            score_val_all = score_val_all + dis_val

    # global NN
    area_train_all_numpy = np.concatenate(area_train_all, axis=1)
    color_train_all_numpy = np.concatenate(color_train_all, axis=1)
    train_global = np.concatenate((area_train_all_numpy, color_train_all_numpy), axis=1)
    nn_train_global = NearestNeighbors(n_neighbors=knn)
    nn_train_global.fit(train_global)

    # test splits
    score_tegood_all = test_global_info(score_tegood_all, area_tegood_all, color_tegood_all, nn_train_global)
    score_telogical_all = test_global_info(score_telogical_all, area_tlogical_all, color_tlogical_all, nn_train_global)
    score_testru_all = test_global_info(score_testru_all, area_stru_all, color_stru_all, nn_train_global)
    # NEW: val
    score_val_all = test_global_info(score_val_all, area_val_all, color_val_all, nn_train_global)

    # labels
    gt_test_good = [False for _ in range(score_tegood_all.shape[0])]
    gt_test_logical = [True  for _ in range(score_telogical_all.shape[0])]
    gt_test_stru = [True     for _ in range(score_testru_all.shape[0])]

    # concat for metrics (기존 그대로)
    all_socres_onlylo = np.concatenate((score_tegood_all, score_telogical_all), axis=0).astype(np.float32)
    all_socres_onlystru = np.concatenate((score_tegood_all, score_testru_all), axis=0).astype(np.float32)

    all_labels_onlog = gt_test_good + gt_test_logical
    all_labels_onlystu = gt_test_good + gt_test_stru

    auroc_alllog = compute_imagewise_retrieval_metrics(all_socres_onlylo, all_labels_onlog)['auroc']
    auroc_onlystu = compute_imagewise_retrieval_metrics(all_socres_onlystru, all_labels_onlystu)['auroc']
    print(f'{classname} auroc_logical: {auroc_alllog} auroc_stru: {auroc_onlystu}')

    class_num += 1
    auroc_log += auroc_alllog
    auroc_stru += auroc_onlystu

# 평균
auroc_log = auroc_log / class_num
auroc_stru = auroc_stru / class_num
print(f'average logical {auroc_log}, average stru {auroc_stru}')
