import os 
import pandas as pd
from tqdm import tqdm 
import json 

def data_check(): 
    csv_path = '/maindata/data/shared/public/aigame/xujing/all_preprocess_results/training_data_600w_filterred_clean_captioned_summary_with_action_balanced.csv' 
    chunksize = 10000
    data_list = []
    count = 0 

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        # print(chunk) 
        count += chunksize 

        for index, row in chunk.iterrows(): 
            # print(row['path'])
            # print(row['text']) 
            video_path = row['path'].rsplit('/', 1)[0]
            # print(row)
            if "InternVId" in video_path:
                continue 
            # print(video_path) 
            file_len = len(os.listdir(video_path))
            if file_len > 2: 
                # print(path_list)
                # print(video_path) 
                # break 
                data = {
                    "video_path": row['path'],
                    "text": row['text']
                }
                data_list.append(data)
            #break 
            if len(data_list) % 500 == 0:
                print(len(data_list)) 

            if len(data_list) % 10000 == 0: 
                with open("video.json", 'w') as f:
                    json.dump(data_list, f)  
                print("write: ", len(data_list)) 
                # break 
        #if len(data_list) > 10000: 
        #    break 
        # break 

import os 
def data_check_ytb(): 
    fold_path =  "/maindata/data/shared/public/chunli.peng/data_to_tag" 
    file_list = os.listdir(fold_path) 
    for file in file_list: 
        if "bilibili_" not in file: 
            continue 
        csv_path = os.path.join(fold_path, file)
        chunksize = 10000
        data_list = []
        count = 0 

        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            # print(chunk) 
            count += chunksize 

            for index, row in chunk.iterrows(): 
                # print(row['path'])
                # print(row['text']) 
                video_path = row['path'].rsplit('/', 1)[0]
                # print(row)
                # print(video_path) 
                file_len = len(os.listdir(video_path))
                if file_len > 2: 
                    # print(path_list)
                    # print(video_path) 
                    # break 
                    data = {
                        "video_path": row['path'],
                        "text": row['text']
                    }
                    data_list.append(data)
                #break 
                if len(data_list) % 500 == 0:
                    print(len(data_list)) 

                if len(data_list) % 10000 == 0: 
                    with open("video_ytb.json", 'w') as f:
                        json.dump(data_list, f)  
                    print("write: ", len(data_list)) 


import os 
def data_check_ytb_fast(): 
    fold_path =  "/maindata/data/shared/public/chunli.peng/data_to_tag" 
    file_list = os.listdir(fold_path) 

    for file in file_list: 
        if "youtube_" not in file: 
            continue 
        csv_path = os.path.join(fold_path, file) 
        print(csv_path) 
        json_path = csv_path.split('/')[-1].rsplit('.', 1)[0] + '.json' 
        print(json_path)
        # return 

        chunksize = 10000
        count = 0 
        tmp_dict = {} 
        name_set = set()
        
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            # print(chunk) 
            count += chunksize 
            print(count) 
            for index, row in chunk.iterrows(): 
                # print(row['path'])
                # print(row['text']) 
                video_path = row['path'].rsplit('/', 1)[0] 
                data = {
                        "video_path": row['path'],
                        "text": row['text']
                    }
                if video_path in name_set: 
                    tmp_dict[video_path].append(
                        data
                    )
                else:
                    tmp_dict[video_path] = [data]
                    name_set.add(video_path)
                # print(row)
                # print(video_path) 
                # file_len = len(os.listdir(video_path))
                # if file_len > 2: 
                    # print(path_list)
                    # print(video_path) 
                    # break 
                #    data = {
                #        "video_path": row['path'],
                #        "text": row['text']
                #    }
                #     data_list.append(data)
                #break 
                # if len(data_list) % 500 == 0:
                #     print(len(data_list)) 

                #if len(data_list) % 10000 == 0: 
                #    with open("video_ytb_fast.json", 'w') as f:
                #        json.dump(data_list, f)  
                #    print("write: ", len(data_list)) 
            # break 
        new_dict = {}
        for k, v in tmp_dict.items():
            name_set = set()
            for n in v:
                name_set.add(n["video_path"]) 
            if len(name_set) > 2: 
                new_dict[k] = v
        print(len(new_dict))
        with open(os.path.join('data', json_path), 'w') as f:
            json.dump(new_dict, f)
        # break

def data_connect(): 
    from tqdm import tqdm 
    with open("video_ytb.json", 'r') as f: 
        data_list = json.load(f) 
    
    data_dict = {} 
    repeat_list = set()
    for data in tqdm(data_list): 
        if data['video_path'] in repeat_list:
            continue
        else:
            repeat_list.add(data['video_path'])
        video_path = data['video_path'].rsplit('/', 1)[0] 
        if video_path in data_dict.keys(): 
            data_dict[video_path].append(data) 
        else:
            data_dict[video_path] = [data] 
    print(len(data_dict.keys()))
    
    # remove only one video 
    new_data_dict = {}
    for k in data_dict.keys(): 
        if len(data_dict[k]) > 2:
            new_data_dict[k] = data_dict[k] 

    print(len(new_data_dict))
    with open("video_dict_ytb.json", 'w') as f:
        json.dump(new_data_dict, f)  



def test_dataset(): 
    from training.dataset import MultiVideoDataset 
    from transformers import AutoTokenizer 
    pretrained_model_name_or_path = '/maindata/data/shared/public/multimodal/share/zhengcong.fei/ckpts/CogVideoX-5b'
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    data_path = "video_dict.json"
    dataset = MultiVideoDataset(data_path, tokenizer,) 
    #dataset[0]  
    print(dataset[1][2])
    for data in dataset:
        text = data[2]
        if 'human' in text: 
            print(text)
            # break 


def test_ckpts(): 
    import torch 
    state_dict = torch.load("in_context_video/tmp.pt")
    for k, v in state_dict.items():
        print(k, v.size())


import json 
def dataset_combine(): 
    # file_list = ['video_dict_ytb.json', 'video_dict.json'] 
    path_file = os.listdir("data")
    new_data_list = {}
    for file in path_file: 
        file_path = os.path.join("data", file,)
        with open(file_path,) as f:
            data_list = json.load(f)
            # new_data_list = dict(new_data_list, **data_list)
            new_data_list.update(data_list)
            print(len(data_list.keys()))
    print(len(new_data_list.keys())) 
    with open("refine_comb_ytb.json", 'w') as f:
        json.dump(new_data_list, f)  


# data_check_ytb_fast() 
# data_connect() 
# test_dataset()
# test_ckpts()
dataset_combine() 