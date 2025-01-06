import json 

def data_formation(): 
    data_path = "/maindata/data/shared/public/zhengcong.fei/code/incontext/people_data_1.json"
    with open(data_path, 'r') as f: 
        data_dict = json.load(f)
    print(len(data_dict["0"])) 
    print(len(data_dict.keys()))
    # print(data_dict["0"])
    # for k in data_dict.keys():
    #    print(len(data_dict["0"]))
    #    break

data_formation() 