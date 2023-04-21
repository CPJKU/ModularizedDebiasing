import subprocess

GPU_ID = 0
N = 5
MODEL_NAME = "bert-base-uncased" # "bert-base-uncased" # "google/bert_uncased_L-4_H-256_A-4" # "google/bert_uncased_L-2_H-128_A-2"
#FIXMASK_PCT_HIGH = 0.1
#FIXMASK_PCT_LOW = 0.05
DS = "pan16"
#PROT_KEY_IDX = None
CMD_IDX = [9,10] #[3] #[9,10] #[1,4,5,6,11,12] #[7,8] # [0] [2] [3]
DEBUG = False

#prot_key_idx = "" if PROT_KEY_IDX is None else f" --prot_key_idx={PROT_KEY_IDX}"
debug = " --debug" if DEBUG else ""
for i in range(N):
    cmds = {
        #0: f"python3 main.py --baseline --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --ds={DS}" + debug,
        #7: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --ds={DS} --prot_key_idx={0}" + debug,
        #8: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --ds={DS} --prot_key_idx={1}" + debug,
        #2: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --ds={DS}" + debug,
        #3: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --adapter --ds={DS}"  + debug,
        9: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --adapter --ds={DS} --prot_key_idx={0}"  + debug,
        10: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --adapter --ds={DS} --prot_key_idx={1}"  + debug,
        #1: f"python3 main.py --baseline --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --adapter --ds={DS}" + debug, 
        #4: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --prot_adapter  --ds={DS} --prot_key_idx={0}" + debug,
        #5: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --prot_adapter  --ds={DS} --prot_key_idx={1}" + debug,
        #6: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --adapter_fusion --ds={DS} " + debug,
        #11: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --adapter_fusion --ds={DS} --prot_key_idx={0}" + debug,
        #12: f"python3 main.py --baseline --adv --gpu_id={GPU_ID} --seed={i} --model_name={MODEL_NAME} --adapter_fusion --ds={DS} --prot_key_idx={1}" + debug
    }
    for j in CMD_IDX:
        subprocess.call(cmds[j], shell=True)