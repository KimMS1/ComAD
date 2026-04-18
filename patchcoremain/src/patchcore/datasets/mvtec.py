# import os
# from enum import Enum
#
# import PIL
# import torch
# from torchvision import transforms
#
# # _CLASSNAMES = [
# #     "bottle",
# #     "cable",
# #     "capsule",
# #     "carpet",
# #     "grid",
# #     "hazelnut",
# #     "leather",
# #     "metal_nut",
# #     "pill",
# #     "screw",
# #     "tile",
# #     "toothbrush",
# #     "transistor",
# #     "wood",
# #     "zipper",
# # ]
# _CLASSNAMES = ['breakfast_box', 'juice_bottle', 'screw_bag', 'pushpins', 'splicing_connectors']
#
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
#
#
# class DatasetSplit(Enum):
#     TRAIN = "train"
#     VAL = "val"
#     TEST = "test"
#
#
# class MVTecDataset(torch.utils.data.Dataset):
#     """
#     PyTorch Dataset for MVTec.
#     """
#
#     def __init__(
#         self,
#         source,
#         classname,
#         resize=256,
#         imagesize=224,
#         split=DatasetSplit.TRAIN,
#         train_val_split=1.0,
#         **kwargs,
#     ):
#         """
#         Args:
#             source: [str]. Path to the MVTec data folder.
#             classname: [str or None]. Name of MVTec class that should be
#                        provided in this dataset. If None, the datasets
#                        iterates over all available images.
#             resize: [int]. (Square) Size the loaded image initially gets
#                     resized to.
#             imagesize: [int]. (Square) Size the resized loaded image gets
#                        (center-)cropped to.
#             split: [enum-option]. Indicates if training or test split of the
#                    data should be used. Has to be an option taken from
#                    DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
#                    mvtec.DatasetSplit.TEST will also load mask data.
#         """
#         super().__init__()
#         self.source = source
#         self.split = split
#         self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
#         self.train_val_split = train_val_split
#
#         self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
#
#         self.transform_img = [
#             transforms.Resize(resize),
#             transforms.CenterCrop(imagesize),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#         ]
#         self.transform_img = transforms.Compose(self.transform_img)
#
#         self.transform_mask = [
#             transforms.Resize(resize),
#             transforms.CenterCrop(imagesize),
#             transforms.ToTensor(),
#         ]
#         self.transform_mask = transforms.Compose(self.transform_mask)
#
#         self.imagesize = (3, imagesize, imagesize)
#
#     def __getitem__(self, idx):
#         classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
#         image = PIL.Image.open(image_path).convert("RGB")
#         image = self.transform_img(image)
#
#         if self.split == DatasetSplit.TEST and mask_path is not None:
#             mask = PIL.Image.open(mask_path)
#             mask = self.transform_mask(mask)
#         else:
#             mask = torch.zeros([1, *image.size()[1:]])
#
#         return {
#             "image": image,
#             "mask": mask,
#             "classname": classname,
#             "anomaly": anomaly,
#             "is_anomaly": int(anomaly != "good"),
#             "image_name": "/".join(image_path.split("/")[-4:]),
#             "image_path": image_path,
#         }
#
#     def __len__(self):
#         return len(self.data_to_iterate)
#
#     def get_image_data(self):
#         imgpaths_per_class = {}
#
#         for classname in self.classnames_to_use:
#             classpath = os.path.join(self.source, classname, self.split.value)
#
#             # 'val'처럼 비어있는 폴더가 있을 경우 건너뛰기
#             if not os.path.exists(classpath):
#                 continue
#
#             anomaly_types = os.listdir(classpath)
#
#             imgpaths_per_class[classname] = {}
#
#             for anomaly in anomaly_types:
#                 anomaly_path = os.path.join(classpath, anomaly)
#
#                 # 'good', 'logical_anomalies' 등은 폴더여야 함
#                 if not os.path.isdir(anomaly_path):
#                     continue
#
#                 anomaly_files = sorted(os.listdir(anomaly_path))
#                 imgpaths_per_class[classname][anomaly] = [
#                     os.path.join(anomaly_path, x) for x in anomaly_files
#                     if os.path.isfile(os.path.join(anomaly_path, x))  # 파일만 리스트에 추가
#                 ]
#
#                 if self.train_val_split < 1.0:
#                     n_images = len(imgpaths_per_class[classname][anomaly])
#                     train_val_split_idx = int(n_images * self.train_val_split)
#                     if self.split == DatasetSplit.TRAIN:
#                         imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
#                                                                      classname
#                                                                  ][anomaly][:train_val_split_idx]
#                     elif self.split == DatasetSplit.VAL:
#                         imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
#                                                                      classname
#                                                                  ][anomaly][train_val_split_idx:]
#
#         # Unrolls the data dictionary to an easy-to-iterate list.
#         data_to_iterate = []
#         for classname in sorted(imgpaths_per_class.keys()):
#             for anomaly in sorted(imgpaths_per_class[classname].keys()):
#                 for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
#                     data_tuple = [classname, anomaly, image_path]
#
#                     if self.split == DatasetSplit.TEST and anomaly != "good":
#
#                         # === MVTec LOCO 경로 수정을 위한 로직 ===
#
#                         # 1. 'test'를 'ground_truth'로 기본 교체
#                         # (결과 예: ...\ground_truth\logical_anomalies\001.png)
#                         base_mask_path = image_path.replace(
#                             os.sep + self.split.value + os.sep,  # \test\
#                             os.sep + "ground_truth" + os.sep  # \ground_truth\
#                         )
#
#                         # 2. 경로와 파일명을 분리
#                         # (dir_part 예: ...\ground_truth\logical_anomalies)
#                         # (image_filename_with_ext 예: 001.png)
#                         dir_part = os.path.dirname(base_mask_path)
#                         image_filename_with_ext = os.path.basename(base_mask_path)
#
#                         # 3. 파일명에서 확장자를 제거 (중간 폴더명 생성)
#                         # (image_filename_no_ext 예: 001)
#                         image_filename_no_ext = os.path.splitext(image_filename_with_ext)[0]
#
#                         # 4. LOCO 경로 구조에 맞게 재조립 (최종 파일명 '000.png' 고정)
#                         # (예: ...\logical_anomalies + \001\ + 000.png)
#                         mask_path = os.path.join(
#                             dir_part,
#                             image_filename_no_ext,
#                             "000.png"  # <-- 요청하신 대로 최종 파일명 고정
#                         )
#                         # === 로직 수정 완료 ===
#
#                         data_tuple.append(mask_path)
#                     else:
#                         # 'good' 샘플이거나 'train' 스플릿인 경우
#                         data_tuple.append(None)
#                     data_to_iterate.append(data_tuple)
#
#         return imgpaths_per_class, data_to_iterate
#     #
#     # def get_image_data(self):
#     #     imgpaths_per_class = {}
#     #     maskpaths_per_class = {}
#     #
#     #     for classname in self.classnames_to_use:
#     #         classpath = os.path.join(self.source, classname, self.split.value)
#     #         maskpath = os.path.join(self.source, classname, "ground_truth")
#     #         anomaly_types = os.listdir(classpath)
#     #
#     #         imgpaths_per_class[classname] = {}
#     #         maskpaths_per_class[classname] = {}
#     #
#     #         for anomaly in anomaly_types:
#     #             anomaly_path = os.path.join(classpath, anomaly)
#     #             anomaly_files = sorted(os.listdir(anomaly_path))
#     #             imgpaths_per_class[classname][anomaly] = [
#     #                 os.path.join(anomaly_path, x) for x in anomaly_files
#     #             ]
#     #
#     #             if self.train_val_split < 1.0:
#     #                 n_images = len(imgpaths_per_class[classname][anomaly])
#     #                 train_val_split_idx = int(n_images * self.train_val_split)
#     #                 if self.split == DatasetSplit.TRAIN:
#     #                     imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
#     #                         classname
#     #                     ][anomaly][:train_val_split_idx]
#     #                 elif self.split == DatasetSplit.VAL:
#     #                     imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
#     #                         classname
#     #                     ][anomaly][train_val_split_idx:]
#     #
#     #             if self.split == DatasetSplit.TEST and anomaly != "good":
#     #                 anomaly_mask_path = os.path.join(maskpath, anomaly)
#     #                 anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
#     #                 maskpaths_per_class[classname][anomaly] = [
#     #                     os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
#     #                 ]
#     #             else:
#     #                 maskpaths_per_class[classname]["good"] = None
#     #
#     #     # Unrolls the data dictionary to an easy-to-iterate list.
#     #     data_to_iterate = []
#     #     for classname in sorted(imgpaths_per_class.keys()):
#     #         for anomaly in sorted(imgpaths_per_class[classname].keys()):
#     #             for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
#     #                 data_tuple = [classname, anomaly, image_path]
#     #                 if self.split == DatasetSplit.TEST and anomaly != "good":
#     #                     data_tuple.append(maskpaths_per_class[classname][anomaly][i])
#     #                 else:
#     #                     data_tuple.append(None)
#     #                 data_to_iterate.append(data_tuple)
#     #
#     #     return imgpaths_per_class, data_to_iterate
import os
from enum import Enum
import PIL
import torch
from torchvision import transforms
import numpy as np  # (np는 ComAD가 사용할 수 있으니 임포트)

# (클래스 이름은 LOCO용으로 이미 잘 수정되어 있습니다)
_CLASSNAMES = ['breakfast_box', 'juice_bottle', 'screw_bag', 'pushpins', 'splicing_connectors']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    (PatchCore와 ComAD 입력을 모두 로드하도록 수정됨)
    """

    def __init__(
            self,
            source,
            classname,
            resize=256,
            imagesize=224,
            split=DatasetSplit.TRAIN,
            train_val_split=1.0,
            **kwargs,
    ):
        """
        Args:
            (기존과 동일)
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        # --- PatchCore용 변환 (기존 코드) ---
        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        # --- ComAD의 'image1'용 변환 (새로 추가) ---
        # (ComAD 로더가 사용하던 ToTensor()와 동일)
        self.transform_to_tensor = transforms.ToTensor()

        # --- 마스크용 변환 (기존 코드) ---
        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    # === __getitem__ 함수 (핵심 수정) ===
    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]

        # 원본 PIL 이미지를 한 번만 엽니다.
        image_pil = PIL.Image.open(image_path).convert("RGB")

        # --- 1. PatchCore용 입력 (기존 'image') ---
        image = self.transform_img(image_pil)

        # --- 2. ComAD용 입력 (ComAD의 'image1'에 해당) ---
        comad_image1 = self.transform_to_tensor(image_pil)
        # (참고: ComAD의 'image'는 PatchCore의 'image'와 동일하므로
        #  run_patchcore.py에서 'image'를 두 모델에 같이 쓰면 됩니다.)

        # --- 3. 마스크 로드 (기존 코드) ---
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        # === 4. 두 모델의 입력을 모두 반환 ===
        return {
            "image": image,  # PatchCore 입력 (ComAD의 'image'로도 사용)
            "comad_image1": comad_image1,  # ComAD의 'image1' 입력
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    # === get_image_data 함수 (LOCO용으로 이미 수정된 상태) ===
    def get_image_data(self):
        imgpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)

            # 'val'처럼 비어있는 폴더가 있을 경우 건너뛰기
            if not os.path.exists(classpath):
                continue

            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)

                # 'good', 'logical_anomalies' 등은 폴더여야 함
                if not os.path.isdir(anomaly_path):
                    continue

                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                    if os.path.isfile(os.path.join(anomaly_path, x))  # 파일만 리스트에 추가
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][train_val_split_idx:]

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]

                    if self.split == DatasetSplit.TEST and anomaly != "good":

                        # === MVTec LOCO 경로 수정을 위한 로직 ===
                        base_mask_path = image_path.replace(
                            os.sep + self.split.value + os.sep,  # \test\
                            os.sep + "ground_truth" + os.sep  # \ground_truth\
                        )
                        dir_part = os.path.dirname(base_mask_path)
                        image_filename_with_ext = os.path.basename(base_mask_path)
                        image_filename_no_ext = os.path.splitext(image_filename_with_ext)[0]
                        mask_path = os.path.join(
                            dir_part,
                            image_filename_no_ext,
                            "000.png"  # 최종 파일명 고정
                        )
                        # === 로직 수정 완료 ===

                        data_tuple.append(mask_path)
                    else:
                        # 'good' 샘플이거나 'train' 스플릿인 경우
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate