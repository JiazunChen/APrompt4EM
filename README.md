# APrompt4EM

This is the code for  "APrompt4EM: Augmented Prompt for GEM with Contextualized Soft Token and Information Supplementation," based on the source codes from [PromptEM](https://github.com/ZJU-DAILY/PromptEM).

## Quick Start

### Dependencies

This project is developed and tested on Ubuntu 20.04.6 LTS with Python 3.8.5. To see additional Python library dependencies, refer to the `requirements.txt` file.

We recommend using Anaconda to create a new environment:
```
conda create -n APrompt4EM python=3.8.5
conda activate APrompt4EM
pip install -r requirements.txt
```
Please note that you **do not** need to install `OpenPrompt` using `pip` manually because we have modified parts of the code. Our implementation of the Contextualized Soft Token Model can be found in `./openprompt/prompts/ptuning_prompts.py`.



### Datasets

We utilize twelve real-world benchmark datasets with varying structures, sourced from [Machamp](https://github.com/megagonlabs/machamp) and [WDC](https://webdatacommons.org/largescaleproductcorpus/wdc-products/).

The processed natural language datasets (including 4 gpt augmented datasets) are stored in `./data/natural/`. Before first use, it is necessary to unzip them:
```
cd ./data/natural
unzip natural.zip
```

## Run

Default parameters (N=0,K=4) are provided for quick and easy testing across all datasets:

```
python main.py --default
```

Additionally, a script is available to facilitate testing with different hyperparameters:
```
python run_all.py
```


By default, the best models are saved in the `./result_model/{time}` directory.

