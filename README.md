# StomAI application
## About


## Installation
<details>
    <summary> Windows Only Pre-Download Steps </summary>

Open command prompt as administrator and run `wsl --install` to enable Windows subsytem for linux. After this completes windows will need to restart.
Visit the Microsoft Store and install Ubuntu 20.04 LTS. Once the store has downloaded the application launch Ubuntu 20.04 LTS from the start menu.
The first time you run the Ubuntu app it will need sometime to setup and ask you to create a username and password to use in the application.
Install the required pre-requist packages by running the following commands:
```
sudo apt update
sudo apt install python3-pip
```
To ensure the python packages installed are accessable from the command line run:
```
echo -e "\nexport PATH=/home/$USER/.local/bin:\$PATH" >> ~/.bashrc
exec bash
```
</details>
<br/>

### Download SAI
To checkout the application to run locally on your machine run the commands:
```
git clone https://github.com/XDynames/SAI-app.git
cd SAI-app
```

### Install
SAI can either use a computer's CPU to process samples or its GPU. The GPU will be significantly faster but requires some extra installation steps:

<details>
    <summary>CPU only mode</summary>
    
    Ensure you have libgeos installed: `sudo apt install libgeos-dev`
    Run `bash setup.sh`
</details>

<details>
    <summary>GPU accelerated mode</summary>

Install the appropriate versions of [Pytorch](https://pytorch.org/get-started/locally/) and [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to suit your GPU.
Run `bash setup_gpu.sh` to install the remaining dependencies.
</details>

## Running SAI
Run `./SAI` while inside the checkouted folder. This will automatically open a webpage in your browser where you can view and use the application

## Referencing
If you use SAI as part of your work please cite us as:
