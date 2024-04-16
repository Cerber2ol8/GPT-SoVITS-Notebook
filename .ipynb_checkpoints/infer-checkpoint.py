import os
import re
from subprocess import Popen
import psutil
import signal
from config import python_exec

from tools.i18n.i18n import I18nAuto
import argparse
i18n = I18nAuto()

parser = argparse.ArgumentParser()

parser.add_argument('--input_text', help='TTS输入文本')

parser.add_argument('--output_path', help='输出路径', 
                    default="output.wav")

parser.add_argument('--ref_wav_path', help='提示音频的路径', 
                    default="data/leijun/leijun_Vocals.wav_0000041600_0000226240.wav")

parser.add_argument('--prompt_text', help='提示文本的路径', 
                    default="一九八七年，我呢考上了武汉大学的计算机系。")

parser.add_argument('--prompt_language', help='提示音频的语言', 
                    default="all_zh")

parser.add_argument('--text_language', help='输入文本的语言')

parser.add_argument('--gpt_path', help='输入文本的语言', 
                    default="train-01-e6.ckpt")

parser.add_argument('--sovits_path', help='输入文本的语言', 
                    default="train-01_e10_s200.pth")

args = parser.parse_args()


pretrained_sovits_name="GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

SoVITS_weight_root="SoVITS_weights"
GPT_weight_root="GPT_weights"

gpt_path = args.gpt_path
sovits_path = args.sovits_path

is_half = False
gpu_number = "0"
if_tts=True


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names,key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names,key=custom_sort_key), "__type__": "update"}


def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):SoVITS_names.append(name)
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append(name)
    return SoVITS_names,GPT_names

def kill_proc_tree(pid, including_parent=True):  
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass

def change_tts_inference(if_tts,bert_path,cnhubert_base_path,gpu_number,gpt_path,sovits_path):
    if(if_tts==True):
        os.environ["gpt_path"]=gpt_path if "/" in gpt_path else "%s/%s"%(GPT_weight_root,gpt_path)
        os.environ["sovits_path"]=sovits_path if "/"in sovits_path else "%s/%s"%(SoVITS_weight_root,sovits_path)
        os.environ["cnhubert_base_path"]=cnhubert_base_path
        os.environ["bert_path"]=bert_path
        os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_number
        os.environ["is_half"]=str(is_half)
        os.environ["input_text"] = args.input_text
        os.environ["output_path"] = args.output_path
        os.environ["ref_wav_path"] = args.ref_wav_path 
        os.environ["prompt_text"] = args.prompt_text
        os.environ["prompt_language"] = args.prompt_language
        os.environ["text_language"] = args.text_language

        cmd = '"%s" GPT_SoVITS/inference_cli.py'%(python_exec)
        print(i18n("TTS推理进程已开启"))
        print(cmd)
        p_tts_inference = Popen(cmd, shell=True)
        p_tts_inference.wait()
        print(i18n("TTS推理进程结束"))
 

if __name__ == '__main__':
    
    change_tts_inference(if_tts,bert_path,cnhubert_base_path,gpu_number,gpt_path,sovits_path)


