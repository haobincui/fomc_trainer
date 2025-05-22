
## Setup
```bash
conda create -n fomc_trainer python=3.10

# for mac and windows
# conda activate fomc_trainer

# for linux
source activate fomc_trainer

pip install -e.[dev]
pip install flash-attn==2.5.6 --no-build-isolation
```


## setup for judge model (GRPO)
```bash
conda create -n vllm_env
# for mac and windows
# conda activate vllm_env

# for linux
source activate vllm_env

pip install "bitsandbytes>=0.43.0"
pip install "peft>=0.14.0"
pip install vllm
```


## run training
```bash
# SFT
bash run_sft.sh

# GRPO
## start the judge model
bash start_vllm.sh

## run training
bash run_grpo.sh
```

## Config Files

Accelerate configs: configs/accelerate/*.yaml

SFT configs: configs/sft/sft_*.yaml
GRPO configs: configs/grpo/grpo_*.yaml

