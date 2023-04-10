IFCNN
Project page of "IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network, Information Fusion, 54 (2020) 99-118".

Requirements
pytorch=0.4.1
python=3.x
torchvision
numpy
opencv-python
jupyter notebook (optional)
anaconda (suggeted)
Configuration
# Create your virtual environment using anaconda
conda create -n IFCNN python=3.5

# Activate your virtual environment
conda activate IFCNN

# Install the required libraries
conda install pytorch=0.4.1 cuda80 -c pytorch
conda install torchvision numpy jupyter notebook
pip install opencv-python
Usage
# Clone our code
git clone https://github.com/uzeful/IFCNN.git
cd IFCNN/Code

# Remember to activate your virtual enviroment before running our code
conda activate IFCNN

# Replicate our image method on fusing multiple types of images
python IFCNN_Main.py

# Or run code part by part in notebook
jupyter notebook IFCNN_Notebook.ipynb


Citation
If you find this code is useful for your research, please consider to cite our paper. Yu Zhang, Yu Liu, Peng Sun, Han Yan, Xiaolin Zhao, Li Zhang, IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network, Information Fusion, 54 (2020) 99-118.

@article{zhang2020IFCNN,
  title={IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network},
  author={Zhang, Yu and Liu, Yu and Sun, Peng and Yan, Han and Zhao, Xiaolin and Zhang, Li},
  journal={Information Fusion},
  volume={54},
  pages={99--118},
  year={2020},
  publisher={Elsevier}
}
