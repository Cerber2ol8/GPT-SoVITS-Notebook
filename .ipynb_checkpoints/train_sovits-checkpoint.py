import json
import os
from subprocess import Popen

if_save_latest = True
if_save_every_weights = True
save_every_epoch = 2
gpu_numbers = "0"


batch_size = 12
total_epoch = 10
text_low_lr_rate = 4


SoVITS_weight_root="SoVITS_weights"
os.makedirs(SoVITS_weight_root,exist_ok=True)

exp_name = "train-01"
exp_root = "experiments"
os.makedirs(exp_root,exist_ok=True)

pretrained_s2G = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_s2D = "GPT_SoVITS/pretrained_models/s2D488k.pth"

is_half = False


def train(batch_size,total_epoch,
               text_low_lr_rate,if_save_latest,if_save_every_weights,
               save_every_epoch,gpu_numbers,pretrained_s2G,pretrained_s2D):
    global p_train_SoVITS
    with open("GPT_SoVITS/configs/s2.json")as f:
        data=f.read()
        data=json.loads(data)
    s2_dir="%s/%s"%(exp_root,exp_name)
    os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)
    if(is_half==False):
        data["train"]["fp16_run"]=False
        batch_size=max(1,batch_size//2)
    s2_dir="%s/%s"%(exp_root,exp_name)
    os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)

    data["train"]["batch_size"]=batch_size
    data["train"]["epochs"]=total_epoch
    data["train"]["text_low_lr_rate"]=text_low_lr_rate
    data["train"]["pretrained_s2G"]=pretrained_s2G
    data["train"]["pretrained_s2D"]=pretrained_s2D
    data["train"]["if_save_latest"]=if_save_latest
    data["train"]["if_save_every_weights"]=if_save_every_weights
    data["train"]["save_every_epoch"]=save_every_epoch
    data["train"]["gpu_numbers"]=gpu_numbers
    data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
    data["save_weight_dir"]=SoVITS_weight_root
    data["name"]=exp_name
    tmp_config_path="%s/tmp_s2.json"%exp_root
    with open(tmp_config_path,"w")as f:f.write(json.dumps(data))

    cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"'%("python",tmp_config_path)
    print("SoVITS训练开始：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True})
    print(cmd)
    p_train_SoVITS = Popen(cmd, shell=True)
    p_train_SoVITS.wait()
    p_train_SoVITS=None
    print("SoVITS训练完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False})

if __name__ == '__main__':

    train(batch_size,total_epoch,
                text_low_lr_rate,if_save_latest,if_save_every_weights,
                save_every_epoch,gpu_numbers,pretrained_s2G,pretrained_s2D)
