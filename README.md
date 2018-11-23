# Laplacian Smoothing

Implementation of “Implicit Fairing of Irregular Meshes using Diffusion and Curvature Flow”, Desbrun et al., Siggraph ’99

# Testing

1. Clone the repo.
1. Build docker image with
```
cd /path/into/repo
sudo docker build -f DockerNew . -t lpl_smth
```
1. Run a container with
```
sudo docker run -it -v /path/into/repo/:/home/keras/notebook/ --name test_case -p 8888:8888 lpl_smth
```
1. Copy URL from your terminal into your browser. Should be something like:
```
http://127.0.0.1:8888/?token=b25820054d06c8e8aacb24c4c053a5f48b746362b30850e0
```
1. Run test_smoothing.ipynb

Smoothed objects are saved under ```smoothed``` folder.
