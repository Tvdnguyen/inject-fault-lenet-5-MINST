export WORKDIR=./
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
# python src/train_model_lenet5_float.py
python src/quant.py