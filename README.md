#### 1 Prepare the dataset

`prepare.py` corresponds to the **Data Processing** section in the paper. If you want to run `prepare.py` (although we do not recommend it), you need to prepare the following parts:

1. NLI
   1. [D3 Repo](https://github.com/caoyu-noob/D3#2-prepare-models)
   2. [Download Link](https://drive.google.com/file/d/1QnT8V2Yj4Zl2yW2rnQIi2p56I_wbN3Ee/view)

2. COMET
	1. [COMET-BART Repo](https://github.com/allenai/comet-atomic-2020)
	2. [Download Link](https://storage.googleapis.com/ai2-mosaic-public/projects/mosaic-kgs/comet-atomic_2020_BART.zip)

3. PEC dataset
	1. [PEC Repo](https://github.com/zhongpeixiang/PEC)
	2. [Download Link](https://www.dropbox.com/s/9lhdf6iwv61xiao/cleaned.zip?dl=0)
	3. [Hugging Face](https://huggingface.co/datasets/viewer/?dataset=pec&config=all)

4. CoMAE dataset
	1. [CoMAE Repo](https://github.com/chujiezheng/CoMAE)
	2. [Download Link](https://1drv.ms/f/s!Aky8v8NZbQx1qjj0aAr--c33hNHY)

5. [GPT2-small](https://huggingface.co/gpt2)  or  [distilgpt2](https://huggingface.co/distilgpt2)

Place all models in *pre_trained* directory and all data in the *data* directory. Like：

```
train.py
prepare.py
pre_trained
	comet-bart
		......
	distilgpt2
		config.json
		merges.txt
		pytorch_model.bin
		tokenizer.json
		vocab.json
	nli
		......
data
	CoMAE
		test_happy_annotated.txt
		test_offmychest_annotated.txt
		train_happy_annotated.txt
		train_offmychest_annotated.txt
		validation_happy_annotated.txt
		validation_offmychest_annotated.txt
	PEC
		happy
			persona.txt
			train.txt
			valid.txt
			test.txt
		offmychest
			......
```

**Notice**

Running `prepare.py` will take a lot of time, so it is not recommended to run it. For convenience, we provide the processed dataset. [[Google Drive](https://drive.google.com/file/d/1c619SHlMeVfqyC8WFY_CSgiPPsQdzA1i/view?usp=sharing)] [[BaiduNetDisk](https://pan.baidu.com/s/10TDFTGB6XVUpzo43pBZ9gQ?pwd=nvfx )]

#### 2 Train

**Prepare the environment**

```shell
pip install -r requestments.txt
```

> Install nlg-eval by following [nlg-eval repo](https://github.com/Maluuba/nlg-eval)

Put the dataset processed by `prepare.py` into *output* directory and run `python train.py`

#### 3 Generation

Edit *infer.json*, set `ckpt`, and run `python infer.py`



