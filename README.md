# Deep Learning In-Fill for Progressive Isosurface Raycasting
This repo holds the model training code for the TVCG paper, ["Interactive Isosurface Visualization in Memory Constrained Environments Using Deep Learning and Speculative Raycasting"](https://ieeexplore.ieee.org/document/10577555) by Landon Dyken, Will Usher, and Sidharth Kumar. The code here was adapted from the work of ["FoVolNet: Fast Volume Rendering using Foveated Deep Neural Networks"](https://ieeexplore.ieee.org/document/9903564) for our training purposes. 

## Running Locally
Training and inference in this codebase require an NVIDIA GPU with CUDA support, along with conda for dependencies. 
- To begin, 'install.sh' must first be run to install necessary dependencies. 
- For example purposes, sample data is given here for training and inference in the 'data/' folder, split into training and validation sets. To preprocess these datasets and begin training a new model, run 'start_training.sh' after installation. During training, a new model will be created in the 'results/new-model/' folder. Checkpoints will be saved every 5 epochs, and training can be stopped at any point by killing the process (ctrl+c). Training can be continued from the latest checkpoint by running 'continue_training.sh'.
- The model used for the TVCG paper is also included under 'results/noof-ultraminiv12'. To test this model, run 'test_inference.sh' after installation. Output PSNR, SSIM, and images will be created in 'infer/'.
