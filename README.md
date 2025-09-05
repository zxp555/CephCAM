# CephCAM

## Generate & evaluate camouflage texture

### Requirements

All the codes are tested in the following environment:

- Linux (Ubuntu 20.04.5)
- python 3.9.19
- cuda 11.6
- pytorch 1.13.0
- numpy 1.26.4
- pytorch3d 0.7.5

Install the environment by:

```sh
conda env create -f ./attack/environment.yml
```



### Prepare checkpoints

```sh
git clone https://github.com/WongKinYiu/yolov9.git

mkdir checkpoints
wget -P ./checkpoints https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt
```



### Generate & evaluate

Run `./attack/attack.ipynb`. The texture will be automatically evaluated after generation.



## Texture display

For wired screens, connect them to the computer using an HDMI cable and set them to expansion mode. 

For wireless screens, make sure they are on the same LAN as the computer. Open the browser and open `http://<computer_ip>:23333`

```bash
cd screen
conda env create -f ./environment.yml

python main.py

# then run ./main.ipynb
```

