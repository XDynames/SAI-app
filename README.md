# StomaAI application
## About


## Installation (~10 minutes)
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
    <summary>CPU mode</summary>
<details>
	<summary>MacOS</summary>

Ensure you have [Homebrew](https://brew.sh/) setup and install the following packages

	brew install geos gdal
If you have a Macbook that uses an Apple Silicone based CPU run

	pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

</details>
<details>
	<summary>Windows and Linux</summary>

Ensure you have libgeos installed:

	sudo apt install libgeos-dev

</details>

Once the operation system specific steps are completed run:

	bash setup.sh

</details>

<details>
    <summary>GPU accelerated mode</summary>

Install the appropriate versions of [Pytorch](https://pytorch.org/get-started/locally/) and [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to suit your GPU.
Run `bash setup_gpu.sh` to install the remaining dependencies.
</details>

## Running SAI
Run `./SAI` while inside the checkouted folder. This will automatically open a webpage in your browser where you can view and use the application.

When using CPU only measurement time has been measured at 60-6 seconds per image, this is also dependant on image resolution.
In GPU accelorated mode this time drops to 0.8-0.2 seconds per image.

## Tested On
Windows: 10 & 11  
Ubuntu: 20.04 & 18.04  
MacOS: BigSur & Monterey  

## Referencing
If you use SAI as part of your work please cite us:
```
@article {Sai2022.02.07.479482,
	author = {Sai, Na and Bockman, James Paul and Chen, Hao and Watson-Haigh, Nathan and Xu, Bo and Feng, Xueying and Piechatzek, Adriane and Shen, Chunhua and Gilliham, Matthew},
	title = {SAI: Fast and automated quantification of stomatal parameters on microscope images},
	elocation-id = {2022.02.07.479482},
	year = {2022},
	doi = {10.1101/2022.02.07.479482},
	URL = {https://www.biorxiv.org/content/early/2022/02/10/2022.02.07.479482},
	eprint = {https://www.biorxiv.org/content/early/2022/02/10/2022.02.07.479482.full.pdf},
	journal = {bioRxiv}
}
```


