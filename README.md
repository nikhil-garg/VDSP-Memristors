# VDSP 

This is the code for the paper *Unsupervised local learning based on voltage-dependent synaptic plasticity for resistive and ferroelectric synapses*, available [here](https://arxiv.org/abs/2510.25787).

## Getting started

The code was written in Python 3.9.

### Dependencies
- numpy
- tqdm
- matplotlib
- fire
- scikit-learn
- pandas
- numba
- python-mnist


### Installation
Dependencies for each subdirectory can be installed with : 
```
pip3 install -r requirements.txt
```

## Usage

To train and evaluate the VDSP on the `MNIST' with memristive model, run the following command: 
```
python3 memristor_mnist.py
```
Simplified implementation without memristive models : 
```
python3 vdsp_mnist_wta_simplified.py
```
Parametric runs : 
```
python3 memristor_mnist.py run_random_vdsp {seed}
```
