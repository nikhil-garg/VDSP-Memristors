import os
import numpy as np
from mnist import MNIST
from tqdm import tqdm
import matplotlib.pyplot as plt
from fire import Fire
from sklearn.model_selection import train_test_split
import pandas as pd
from numba import njit, vectorize, float32, float64
from memristor_mnist import *

def generate_sample(mid_value, scaling_ratio, seed, param_type='float'):
    np.random.seed(seed)
    low = mid_value * (1 - scaling_ratio)
    high = mid_value * (1 + scaling_ratio)
    if param_type == 'int':
        return np.random.randint(low, high)
    return np.random.uniform(low, high)

def run_random_vdsp(seed):

    normalize_duration = False
    weight_init_scale_max=1
    nb_epochs = 1
    output_file = f"accuracy_{seed}.csv"
    np.random.seed(seed)
    # Fixed parameters
    train_size=60000
    test_size = 10000
    duration_per_sample = 40
    with_plots = False
    refractory_period_input = 1
    input_threshold = 1
    input_leak_ms = 29
    input_scale = 0.2
    input_bias_scale = 1.28
    noise_input = 1
    noise_bias = 0.9793
    presentation_time_multiplier = 4
    input_leak_cst_negative_ratio = 1
    th_leak_cst_ms_scale = 10
    refractory_period= 3
    n_output_neurons = 20

    if seed == 100000:   
        with_plots = True
        device = "tio2"
        memristor_params = get_tio2_params()
        vdsp_lr=memristor_params["lr"]  
        lr_pn=memristor_params["lr_pn"]
        gamma=memristor_params["gamma"]
        gamma_pn=memristor_params["gamma_pn"]
        alpha=memristor_params["alpha"]
        alpha_pn=memristor_params["alpha_pn"]
        vth=memristor_params["vth"]
        vth_pn=memristor_params["vth_pn"]
        wmin = 1 / memristor_params["HRS_LRS_ratio"]

        #Device dependent network parameters
        # pos_mult_init = 1.6613
        # neg_mult_init = 1.47
        pos_mult_init_min = vth*vth_pn
        neg_mult_init_min = vth
        pos_mult_init = pos_mult_init_min*1.11 # SC LTD
        neg_mult_init = neg_mult_init_min*1.05 # SC LTP
        output_leak_ms = 11.51
        th_inc_ratio = 1.1
        lateral_inhibition_period_scale = 4
        output_threshold = 8.2
        duration_per_sample = 40
    if seed == 200000:   
        with_plots = True
        device = "hzo"
        memristor_params = get_hzo_params()
        vdsp_lr=memristor_params["lr"]  
        lr_pn=memristor_params["lr_pn"]
        gamma=memristor_params["gamma"]
        gamma_pn=memristor_params["gamma_pn"]
        alpha=memristor_params["alpha"]
        alpha_pn=memristor_params["alpha_pn"]
        vth=memristor_params["vth"]
        vth_pn=memristor_params["vth_pn"]
        wmin = 1 / memristor_params["HRS_LRS_ratio"]

        #Device dependent network parameters
        # pos_mult_init = 1.6613
        # neg_mult_init = 1.47
        pos_mult_init_min = vth*vth_pn
        neg_mult_init_min = vth
        pos_mult_init = pos_mult_init_min*1.26
        neg_mult_init = neg_mult_init_min*1.05
        output_leak_ms = 11.51
        th_inc_ratio = 1.1
        lateral_inhibition_period_scale = 4
        output_threshold = 8.2
        duration_per_sample = 40

    if seed == 300000:   
        with_plots = True
        device = "hfo2"
        memristor_params = get_hfo2_params()
        vdsp_lr=memristor_params["lr"]  
        lr_pn=memristor_params["lr_pn"]
        gamma=memristor_params["gamma"]
        gamma_pn=memristor_params["gamma_pn"]
        alpha=memristor_params["alpha"]
        alpha_pn=memristor_params["alpha_pn"]
        vth=memristor_params["vth"]
        vth_pn=memristor_params["vth_pn"]
        wmin = 1 / memristor_params["HRS_LRS_ratio"]

        #Device dependent network parameters
        # pos_mult_init = 1.6613
        # neg_mult_init = 1.47
        pos_mult_init_min = vth*vth_pn
        neg_mult_init_min = vth
        pos_mult_init = pos_mult_init_min*1.08
        neg_mult_init = neg_mult_init_min*1.08
        output_leak_ms = 11.51
        th_inc_ratio = 1.1
        lateral_inhibition_period_scale = 4
        output_threshold = 8.2
        duration_per_sample = 40

    nb_exp = 20000    
    if (seed > 0) and (seed < nb_exp): 
        normalize_duration = False  
        n_output_neurons = 50
        with_plots = False
        device = "tio2"
        memristor_params = get_tio2_params()
        vdsp_lr=memristor_params["lr"]  
        lr_pn=memristor_params["lr_pn"]
        gamma=memristor_params["gamma"]
        gamma_pn=memristor_params["gamma_pn"]
        alpha=memristor_params["alpha"]
        alpha_pn=memristor_params["alpha_pn"]
        vth=memristor_params["vth"]
        vth_pn=memristor_params["vth_pn"]
        wmin = 1 / memristor_params["HRS_LRS_ratio"]
        #Device dependent network parameters
        output_leak_ms = 11.51
        th_inc_ratio = 1.1
        lateral_inhibition_period_scale = 4
        output_threshold = 8.2
        pos_mult_init_min = vth*vth_pn
        neg_mult_init_min = vth
        pos_mult_init_array = np.linspace(1.05, 1.5, 16)
        neg_mult_init_array = np.linspace(1.05, 1.5, 16)
        # n_output_neurons_array = [10, 50, 100, 200, 500]
        n_output_neurons_array = [10, 50,200, 500]

        # nb_epochs_array = [1,2,3]
        nb_epochs_array = [1]

        train_size_array = [6000, 12000, 30000, 60000]
        # train_size_array = [60000]

        parametric_grid = np.array(np.meshgrid(pos_mult_init_array, neg_mult_init_array, n_output_neurons_array, nb_epochs_array, train_size_array)).T.reshape(-1, 5)
        pos_mult_init, neg_mult_init, n_output_neurons, nb_epochs, train_size = parametric_grid[seed % len(parametric_grid)]

        pos_mult_init = pos_mult_init*vth*vth_pn
        neg_mult_init = neg_mult_init*vth
        n_output_neurons, nb_epochs, train_size = int(n_output_neurons), int(nb_epochs), int(train_size)   

    
   

    input_leak_cst_ms_negative = input_leak_cst_negative_ratio * input_leak_ms
    input_leak_cst_negative = np.exp(-1 / input_leak_cst_ms_negative)
    input_leak_cst = np.exp(-1 / input_leak_ms)
    output_leak_cst = np.exp(-1 / output_leak_ms)
    lateral_inhibition_period = lateral_inhibition_period_scale * refractory_period
    th_leak_cst_ms = th_leak_cst_ms_scale * output_leak_ms
    th_leak_cst = np.exp(-1 / th_leak_cst_ms)
    th_inc = th_inc_ratio * input_threshold
    ans = run_vdsp(
        seed=seed,
        vdsp_lr=vdsp_lr,
        alpha=alpha,
        gamma=gamma,
        wmin=wmin,
        wmax=1,
        vth=vth,
        vprog=0,
        lr_pn=lr_pn,
        gamma_pn=gamma_pn,
        alpha_pn=alpha_pn,
        vth_pn=vth_pn,
        n_output_neurons=n_output_neurons,
        input_leak_cst=input_leak_cst,
        input_leak_cst_negative=input_leak_cst_negative,
        output_leak_cst=output_leak_cst,
        input_threshold=1,  # np.random.uniform(1.091455601/2, 2*1.091455601)
        input_reset=-1,  # np.random.uniform(1.091455601/2, 2*1.091455601)
        refractory_period_input=refractory_period_input,
        output_threshold=output_threshold,
        weight_init_scale_min=0,
        weight_init_scale_max=weight_init_scale_max,
        input_scale=input_scale,
        input_bias_scale=input_bias_scale,
        nb_epochs=nb_epochs,
        use_vdsp=True,
        device=device,
        with_validation=False,
        th_leak_cst=th_leak_cst,
        th_inc=th_inc,  
        refractory_period=refractory_period,
        lateral_inhibition_period=lateral_inhibition_period,
        nb_states = -1,
        nb_states_scale = 'log',
        quantize_read = False,
        quantize_write = False,
        noise_input = noise_input,
        with_plots = with_plots,
        train_size = train_size,
        test_size = test_size,
        presentation_time_multiplier=presentation_time_multiplier,
        noise_bias = noise_bias,
        normalize_duration = normalize_duration,
        duration_per_sample = duration_per_sample,
        duration_between_samples = 100,
        pos_mult_init = pos_mult_init,
        neg_mult_init = neg_mult_init)

    df = pd.read_csv(output_file)
    df["max_frequency"] = ans[1]
    df["accuracy"] = ans[0]
    df['nb_spikes_out_per_sample'] = ans[2]
    df['duration_per_sample'] = ans[3]
    df['fraction_non_zero'] = ans[4]
    #Write the ratios used in the experiment to the csv file
    df["th_leak_cst_ms_scale"] = th_leak_cst_ms_scale
    df["th_inc_ratio"] = th_inc_ratio
    df["lateral_inhibition_period_scale"] = lateral_inhibition_period_scale
    df["input_leak_cst_negative_ratio"] = input_leak_cst_negative_ratio

    df.to_csv(output_file, index=True)
if __name__ == "__main__":
    Fire()

