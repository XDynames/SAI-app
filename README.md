# Install StomaAI
To install the requirements for the application to run locally on your machine run the commands:
```
git clone https://github.com/XDynames/SAI-app.git
cd SAI-app
```
## CPU only installation
Ensure you have libgeos installed: `sudo apt-get install libgeos-dev`
Run `bash setup.sh`

## GPU installation
Install the appropriate versions of [Pytorch](https://pytorch.org/get-started/locally/) and [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to suit your GPU.
Run `bash setup_gpu.sh` to install the remaining dependencies.

# Run the app
Enter `streamlit run main.py` while inside the checkouted folder. This will automatically open a webpage in your browser where you can view and use the application
