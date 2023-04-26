if [[ "$OSTYPE" == "darwin"* ]]; then
pip3 install torch torchvision torchaudio
else
pip3 install --find-links https://download.pytorch.org/whl/torch_stable.html \
    torch==1.11.0+cpu \
    torchvision==0.12.0+cpu \
    torchaudio==0.11.0+cpu
fi
pip3 install git+https://github.com/facebookresearch/detectron2.git
pip3 install -r requirements.txt
pip3 install mask-to-polygons==0.0.2 --no-deps
pip3 install -e .
