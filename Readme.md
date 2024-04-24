## Relative Importance and Activations (RIA)

**PLUG-AND-PLAY: AN EFFICIENT POST-TRAINING PRUNING METHOD FOR LARGE LANGUAGE MODELS**

Yingtao Zhang<sup>1,2</sup>, Haoli Bai<sup>4</sup>, Haokun Lin<sup>5</sup>, Jialin Zhao<sup>1,2</sup>, Lu Hou<sup>4</sup>, & Carlo Vittorio Cannistraci<sup>1,2,3</sup>

<sup>1</sup> Center for Complex Network Intelligence, Tsinghua Laboratory of Brain and Intelligence  
<sup>2</sup> Department of Computer Science, Tsinghua University  
<sup>3</sup> Department of Biomedical Engineering, Tsinghua University  
<sup>4</sup> Huawei Noah’s Ark Lab  
<sup>5</sup> Institute of Automation, Chinese Academy of Sciences  

Corresponding to {zhangyingtao1024, kalokagathos.agon}@gmail.com



#### Setup

Step 1: Create a new conda environment:

```
conda create -n ria python=3.10
conda activate ria
```



Step 2: Install relevant packages

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu121

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



#### Additional Experimental Results on LLaMA3

------

LLaMA3-8B on Wikitext2: fully connected: PPL 6.14

|                     | 50% unstructured sparsity | 2:4   | 2:4+Channel Permutation | 4:8   |
| ------------------- | ------------------------- | ----- | ----------------------- | ----- |
| Magnitude           | 2499.39                   |       |                         |       |
| Relative Importance | 135.77                    |       |                         |       |
| Wanda               | 10.82                     | 24.18 | 22.03                   |       |
| Sparsegpt           | 9.40                      | 16.26 |                         | 12.13 |
| Ria                 | 9.34                      | 23.08 | 20.05                   |       |



LLaMA3-70B on Wikitext2: fully connected: PPL 2.85

|                     | 50% unstructured sparsity | 2:4  | 2:4+Channel Permutation | 4:8  |
| ------------------- | ------------------------- | ---- | ----------------------- | ---- |
| Magnitude           | 19.11                     |      |                         |      |
| Relative Importance | 6.09                      |      |                         |      |
| Wanda               | 6.56                      | 9.28 |                         |      |
| Sparsegpt           | 5.79                      |      |                         |      |
| Ria                 | 5.49                      | 8.35 |                         |      |



#### Acknowledgment

---

This repository is built upon the [SparseGPT](https://github.com/IST-DASLab/sparsegpt) and [Wanda](https://github.com/locuslab/wanda) repository.



#### Citation

----

If you use our code, please consider to cite:

```
@inproceedings{zhangplug,
  title={Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models},
  author={Zhang, Yingtao and Bai, Haoli and Lin, Haokun and Zhao, Jialin and Hou, Lu and Cannistraci, Carlo Vittorio},
  booktitle={The Twelfth International Conference on Learning Representations}
}
```
