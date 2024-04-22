## Relative Importance and Activations (RIA)

---------------



#### Setup

Step 1: Create a new conda environment:

```
conda create -n ria python=3.10
conda activate ria
```



Step 2: Install relevant packages

```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install transformers=4.36.2 sentencepiece=0.1.99 datasets=2.16.1 bitsandbytes=0.42.0
pip install accelerate=0.26.1
```



Step 3: install lm-evaluation-harness (if `--eval_zero_shot`)

Follow the installation here: https://github.com/EleutherAI/lm-evaluation-harness



#### Usage

RIA with unstructured 50% sparsity

```
python main.py \
	--model YOUR_MODEL_NAME \
	--prune_method ria \
	--sparsity_ratio 0.5 \
	--sparsity_type unstructured \
	--save \
```

Here the prune_method can be replaced with wanda, sparsegpt, ri, magnitude



RIA with semi-structured sparsity 

```
python main.py \
	--model YOUR_MODEL_NAME \
	--prune_method ria \
	--sparsity_ratio 0.5 \
	--sparsity_type 2:4 \
	--save \
```

sparsity_type can be any type of semi-structured sparsity pattern, for instance: 1:4, 2:4.

Enable `--reallocation` if you want to use heuristic channel reallocation.

Enable `--lsa` if you want to further finetune the channels after reallocation with linear sum assignment.

Enable `--fast` if you want to use a fast version of linear sum assignment.



#### End-to-End inference speedup with semi-structured sparsity

--------

Currently, this repo only supports the acceleration after direct N:M sparsity. The acceleration of N:M sparsity after channel permutation is still under testing. 

```
python main.py \
	--model YOUR_MODEL_NAME \
	--prune_method ria \
	--sparsity_ratio 0.5 \
	--sparsity_type 2:4 \
	--semi_sparse_acc \
	--save \
```



Requirements:

- `PyTorch >= 2.1.`
- `A NVIDIA GPU with semi-structured sparsity support (Compute Capability 8.0+).`

Make sure that your GPU support cusparselt, otherwise please set

`SparseSemiStructuredTensor._FORCE_CUTLASS = True`

Which force to use CUTLASS

