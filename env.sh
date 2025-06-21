pip install "transformers[pytorch]"
pip install lion-pytorch

export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download openai-community/gpt2 --local-dir /opt/tiger/vtcl/optimize/model/gpt2
huggingface-cli download tatsu-lab/alpaca --local-dir /opt/tiger/vtcl/optimize/dataset/ --repo-type dataset