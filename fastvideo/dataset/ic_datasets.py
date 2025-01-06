import json
import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader
from os.path import join as opj
from collections import Counter
import random 
import torchvision.transforms as TT 
from torchvision.transforms import InterpolationMode  
from torchvision.transforms.functional import center_crop, resize 
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm
from PIL import Image
from fastvideo.utils.dataset_utils import DecordInit
import torchvision
from fastvideo.utils.logging_ import main_print


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DataSetProg(metaclass=SingletonMeta):
    def __init__(self):
        self.cap_list = []
        self.elements = []
        self.num_workers = 1
        self.n_elements = 0
        self.worker_elements = dict()
        self.n_used_elements = dict()

    def set_cap_list(self, num_workers, cap_list, n_elements):
        self.num_workers = num_workers
        self.cap_list = cap_list
        self.n_elements = n_elements
        self.elements = list(range(n_elements))
        random.shuffle(self.elements)
        print(f"n_elements: {len(self.elements)}", flush=True)

        for i in range(self.num_workers):
            self.n_used_elements[i] = 0
            per_worker = int(math.ceil(len(self.elements) / float(self.num_workers)))
            start = i * per_worker
            end = min(start + per_worker, len(self.elements))
            self.worker_elements[i] = self.elements[start:end]

    def get_item(self, work_info):
        if work_info is None:
            worker_id = 0
        else:
            worker_id = work_info.id

        idx = self.worker_elements[worker_id][
            self.n_used_elements[worker_id] % len(self.worker_elements[worker_id])
        ]
        self.n_used_elements[worker_id] += 1
        return idx


dataset_prog = DataSetProg()


def filter_resolution(h, w, max_h_div_w_ratio=17 / 16, min_h_div_w_ratio=8 / 16):
    if h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio:
        return True
    return False


class Incontext_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer, transform_topcrop):
        self.data = args.data_merge_path 
        print(self.data)
        self.num_frames = args.num_frames
        self.train_fps = args.train_fps
        self.use_image_num = args.use_image_num
        self.transform = transform
        self.transform_topcrop = transform_topcrop
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.text_max_length = args.text_max_length
        self.cfg = args.cfg
        self.speed_factor = args.speed_factor
        self.max_height = args.max_height
        self.max_width = args.max_width
        print("height width: ", self.max_height, self.max_width)
        self.drop_short_ratio = args.drop_short_ratio
        assert self.speed_factor >= 1
        self.v_decoder = DecordInit()
        self.video_length_tolerance_range = args.video_length_tolerance_range
        self.support_Chinese = True
        if not ("mt5" in args.text_encoder_name):
            self.support_Chinese = False

        self.cap_list = self.get_cap_list()
        self.key_list = list(self.cap_list.keys()) * 2
        assert len(self.cap_list.keys()) > 0
        # cap_list, self.sample_num_frames = self.define_frame_index(cap_list)

        n_elements = len(self.key_list) 
        # dataset_prog.set_cap_list(args.dataloader_num_workers, cap_list, n_elements)

        # print(f"video length: {len(dataset_prog.cap_list)}", flush=True)

    def set_checkpoint(self, n_used_elements):
        for i in range(len(dataset_prog.n_used_elements)):
            dataset_prog.n_used_elements[i] = n_used_elements

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx): 
        data = self.get_data(idx)
        return data

    def get_data(self, idx):
        # path = dataset_prog.cap_list[idx]["path"]
        key = self.key_list[idx] 
        return self.get_video(key)

    def _resize_for_rectangle_crop(self, arr):
        image_size = self.max_height // 2, self.max_width // 2
        reshape_mode = "center"
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def get_video(self, key): 
        while True: 
            try:
                video_path_list = random.sample(self.cap_list[key], 4)
                video_clips = []
                text = "Four video storyboards. " 

                for i in range(len(video_path_list)):
                    video_path = video_path_list[i]['video_path']
                    assert os.path.exists(video_path), f"file {video_path} do not exist!"
                    # frame_indices = dataset_prog.cap_list[idx]["sample_frame_index"]
                    torchvision_video, _, metadata = torchvision.io.read_video(
                        video_path, output_format="TCHW"
                    )
                    start_frame_idx = 0
                    frame_interval = 1
                    random_range = torchvision_video.size(0) - frame_interval * self.num_frames - 1
                    random_range = max(1, random_range)
                    start_frame = random.randint(1, random_range) if random_range > 0 else 1
                    
                    frame_indices = np.arange(
                        start_frame, start_frame+frame_interval*self.num_frames, frame_interval
                    ).astype(int)
                    video = torchvision_video[frame_indices] 
                    video = self._resize_for_rectangle_crop(video) 
                    
                    text += " [" + str(i+1) + "] " + video_path_list[i]['text'] + "."
                    video_clips.append(video)
                if len(video_clips) == 4:
                    break
            except:
                continue 

        #print(text)
        top = torch.cat((video_clips[0], video_clips[1]), dim=3)
        bottom = torch.cat((video_clips[2], video_clips[3]), dim=3) 
        video = torch.cat([top, bottom], dim=2)
        # print(video.size())
        video = self.transform(video)
        video = rearrange(video, "t c h w -> c t h w")
        video = video.to(torch.uint8)
        assert video.dtype == torch.uint8
        video = video.float() / 127.5 - 1.0

        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        text = text[0] if random.random() > self.cfg else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_tokens_and_mask["input_ids"]
        cond_mask = text_tokens_and_mask["attention_mask"]
        return dict(
            pixel_values=video,
            text=text,
            input_ids=input_ids,
            cond_mask=cond_mask,
            path=video_path,
        )

    
    def decord_read(self, path, frame_indices):
        decord_vr = self.v_decoder(path)
        video_data = decord_vr.get_batch(frame_indices).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data

    def read_jsons(self, data):
        cap_lists = dict()
        with open(data, "r") as f:
            folder_anno = [
                i.strip().split(",") for i in f.readlines() if len(i.strip()) > 0
            ]
        # print(folder_anno)
        for folder, anno in folder_anno:
            print(anno)
            with open(anno, "r") as f:
                sub_list = json.load(f)
            cap_lists.update(sub_list)
        return cap_lists

    def get_cap_list(self):
        cap_lists = self.read_jsons(self.data)
        return cap_lists
