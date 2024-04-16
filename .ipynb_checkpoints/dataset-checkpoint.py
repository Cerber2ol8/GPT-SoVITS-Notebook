import json
import os
from subprocess import Popen
from tools import my_utils
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix,is_share
import traceback

inp_text = "data/leijun_asr/leijun.list"
inp_wav_dir = "data/leijun"
gpu_id = "0"




exp_name = "train-01"
exp_root = "experiments"
os.makedirs(exp_root,exist_ok=True)


pretrained_s2G = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_s2D = "GPT_SoVITS/pretrained_models/s2D488k.pth"
bert_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/"
ssl_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"

ps1abc = []


def setup(inp_text,inp_wav_dir,gpu_id,
          bert_pretrained_dir,ssl_pretrained_dir,pretrained_s2G):
    global ps1abc
    gpu_numbers1a = gpu_id
    gpu_numbers1Ba = gpu_id
    gpu_numbers1c = gpu_id

    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if (ps1abc == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        try:
            #############################1a
            path_text="%s/2-name2text.txt" % opt_dir
            if(os.path.exists(path_text)==False or (os.path.exists(path_text)==True and len(open(path_text,"r",encoding="utf8").read().strip("\n").split("\n"))<2)):
                config={
                    "inp_text":inp_text,
                    "inp_wav_dir":inp_wav_dir,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "bert_pretrained_dir":bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names=gpu_numbers1a.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                print("进度：1a-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True})
                for p in ps1abc:p.wait()

                opt = []
                for i_part in range(all_parts):#txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")

            print("进度：1a-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True})
            ps1abc=[]
            #############################1b
            config={
                "inp_text":inp_text,
                "inp_wav_dir":inp_wav_dir,
                "exp_name":exp_name,
                "opt_dir":opt_dir,
                "cnhubert_base_dir":ssl_pretrained_dir,
            }
            gpu_names=gpu_numbers1Ba.split("-")
            all_parts=len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    }
                )
                os.environ.update(config)
                cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            print("进度：1a-done, 1b-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True})
            for p in ps1abc:p.wait()
            print("进度：1a1b-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True})
            ps1abc=[]
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if(os.path.exists(path_semantic)==False or (os.path.exists(path_semantic)==True and os.path.getsize(path_semantic)<31)):
                config={
                    "inp_text":inp_text,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "pretrained_s2G":pretrained_s2G,
                    "s2config_path":"GPT_SoVITS/configs/s2.json",
                }
                gpu_names=gpu_numbers1c.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                print("进度：1a1b-done, 1cing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True})
                for p in ps1abc:p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                print("进度：all-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True})
            ps1abc = []
            print("一键三连进程结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False})
        except:
            traceback.print_exc()
            print("一键三连中途报错", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False})
    else:
        print("已有正在进行的一键三连任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True})


if __name__ == '__main__':


    setup(inp_text,inp_wav_dir,gpu_id,
          bert_pretrained_dir,ssl_pretrained_dir,pretrained_s2G)


