import os
import numpy as np
from mnist import MNIST
from tqdm import tqdm
import matplotlib.pyplot as plt
from fire import Fire
from sklearn.model_selection import train_test_split
import pandas as pd

from numba import njit, vectorize, float32, float64


#Plot the membrane potential evolution of a neuron for aa given input value for a given number of samples and for TiO2 network parameters
def plot_membrane_potential(
        X=0.5,
        nbtimesteps=350,):
    
    network_params = get_tio2_network_params()
    input_reset = network_params['input_reset']
    input_threshold = network_params['input_threshold']
    input_leak_cst = network_params['input_leak_cst']
    input_scale = network_params['input_scale']
    refractory_period_input = int(network_params['refractory_period_input'])
    refractory_period_input = 2
    input_bias_scale = network_params['input_bias_scale']

    input_bias = compute_input_bias(input_leak_cst, input_threshold, input_bias_scale)


    mem_pot_input = np.zeros(nbtimesteps)
    input_spike = np.zeros(nbtimesteps)
    scaled_input = X * input_scale
    refractory_flag = 0
    for t in range(nbtimesteps):
        if t==0:
            mem_pot_input[t] = 0
        elif refractory_flag == 1:
            refractory_period_input = refractory_period_input -1
            mem_pot_input[t] = input_reset
            input_spike[t] = input_threshold
            if refractory_period_input == 0:
                refractory_flag = 0
            # t = t + refractory_period_input
        else:
            mem_pot_input[t] = mem_pot_input[t-1] * input_leak_cst + input_bias + scaled_input   
            refractory_period_input = 2
            if mem_pot_input[t] > input_threshold:
                mem_pot_input[t] = input_reset
                input_spike[t] = input_threshold
                refractory_flag = 1

    fig = plt.figure(figsize=(6, 2.5))
    plt.rcParams.update({'font.size': 14})

    ax = fig.add_subplot(111)

    ax.plot(mem_pot_input, label='Membrane potential', color='red')

    ax.plot(input_spike, label='Output spike', color='black')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Input value = ' + str(X))

    #Show legend
    if (X==0):
        ax.legend()
    # plt.show()
    #Save figure
    fig_name = 'membrane_potential_input_' + str(X) + '.svg'

    plt.tight_layout()

    plt.savefig(fig_name, dpi=300)        




@njit
def clip(val, min, max):
    return np.maximum(np.minimum(val, max), min)


@njit
def vdsp_memristor(
    w,
    vmem,
    lr=0.01,  # Learning rate
    lr_pn=1,  # Learning rate asymettry
    gamma=1,  # Non-linearity
    gamma_pn=1,
    alpha=1,  # Exponential response
    alpha_pn=1,
    vth=0.1,  # Switching threshold
    vth_pn=1,
):

    lr_p = lr
    lr_n = lr * lr_pn
    alpha_p = alpha
    alpha_n = alpha * alpha_pn
    vth_p = vth
    vth_n = vth * vth_pn
    gamma_p = gamma
    gamma_n = gamma * gamma_pn
    # gamma_n = (w>0.1)*gamma_n + (w<=0.1)*gamma_p
    # gamma_p = (w<0.9)*gamma_p

    f_p = np.power(1-w,gamma_p)
    # f_n = ((w>0.1) * np.power(w,gamma_n)) + ((w<=0.1) * np.power(0.1,gamma_n))
    f_n = np.power(w,gamma_n)
    f_n = (w>0.01)*f_n + (w<=0.01)
    cond_p = vmem < -vth_p
    cond_n = vmem > vth_n

    g_p = np.abs(np.exp(-alpha_p*(vmem+vth_p)) - 1)
    g_n = np.abs(np.exp(alpha_n*(vmem-vth_n)) - 1)

    dW = (cond_p * f_p * g_p * lr_p) - (cond_n * f_n * g_n * lr_n)

    return dW

# DUT,gamma,alpha,vth,vth_pn,gamma_pn,alpha_pn
# tio2,1.6795809048995716,0.678960430914287,1.4323257535409337,1.0921172768459333,0.943858886222557,1.1247972886952367
# hzo,1.0679036578469285,1.1592090321547137,0.41171077312240245,0.9437852566146342,1.579831911425707,0.4741462553774615
# hfo2,1.0172004060753206,0.9608160663826949,0.8038380759183791,1.0596296170810442,0.2000000009312879,1.3356519108359932
# lr=1, lr_pn=1 for all
def get_tio2_params():
    return {
        'lr': 1,
        'lr_pn': 1,
        'gamma': 1.6795809048995716,
        'gamma_pn': 0.943858886222557,
        'alpha': 0.678960430914287,
        'alpha_pn': 1.1247972886952367,
        'vth': 1.4323257535409337,
        'vth_pn': 1.0921172768459333,
        'HRS_LRS_ratio': 7.5
    }

def get_hzo_params():
    return {
        'lr': 1,
        'lr_pn': 1,
        'gamma': 1.0679036578469285,
        'gamma_pn': 1.579831911425707,
        'alpha': 1.1592090321547137,
        'alpha_pn': 0.4741462553774615,
        'vth': 0.41171077312240245,
        'vth_pn': 0.9437852566146342,
        'HRS_LRS_ratio': 2.65

    }

def get_hfo2_params():
    return {
        'lr': 1,
        'lr_pn': 1,
        'gamma': 1.0172004060753206,
        'gamma_pn': 0.5,
        'alpha': 0.9608160663826949,
        'alpha_pn': 1.3356519108359932,
        'vth': 0.8038380759183791,
        'vth_pn': 1.0596296170810442,
        'HRS_LRS_ratio': 4

    }



def get_default_params():
    return dict(
        lr=0.01,
        lr_pn=1,
        gamma=1,
        gamma_pn=1,
        alpha=1,
        alpha_pn=1,
        vth=0.1,
        vth_pn=1,
    )

def plot_vdsp_memristor():
    tio2_params = get_tio2_params()
    wmin = 0.1
    wmax=1

    weights = np.arange(0, 1, 0.1)
    vmem = np.arange(-1.8, 1.8, 0.1)

    effective_weights = weights * (wmax - wmin) + wmin

    W, V = np.meshgrid(weights, vmem)
    EW, _ = np.meshgrid(effective_weights, vmem)
    DW = vdsp_memristor(W, V, **tio2_params)

    print(DW)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(EW, V, DW)

    print(DW.mean())

    plt.show()


@njit
def find_nearest_value(weights, stable_weights):
    nearest_values = np.zeros_like(weights)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            min_diff = np.inf
            for k in range(stable_weights.shape[0]):
                diff = abs(weights[i, j] - stable_weights[k])
                if diff < min_diff:
                    min_diff = diff
                    nearest_val = stable_weights[k]
            nearest_values[i, j] = nearest_val
    return nearest_values

# @njit
def quantize_linear(weights, nb_states):
    return np.round(weights * (nb_states - 1)) / (nb_states - 1)

# @njit
def quantize_log(weights, nb_states):
    return (np.exp(np.round(nb_states*weights)/nb_states)-1)/(np.e-1)


# @njit
def run_one_sample(
    X,
    mem_pot_input,
    mem_pot_output,
    weights,
    dW_pot_history,
    dW_dep_history,
    sample_history,
    duration_per_sample,
    duration_between_samples,
    input_leak_cst,
    input_leak_cst_negative,
    input_bias,
    input_threshold,
    input_reset,
    output_leak_cst,
    refractory_period,
    refractory_period_input,
    ### Lateral inhibition parameters
    lateral_inhibition_period,
    ### VDSP parameters
    use_vdsp,
    vdsp_lr,
    lr_pn,
    gamma,
    gamma_pn,
    alpha,
    alpha_pn,
    vth,
    vth_pn,
    wmin,  # LRS/HRS
    wmax,
    vprog,
    switching_prob,  # Switching probebility
    LRS_mask,
    HRS_mask,
    read_variability,
    write_variability,
    clipen,
    vclipmin,
    vclipmax,
    ### Adaptive threshold parameters (output layer)
    thresholds,
    th_inc,
    th_leak_cst,
    threshold_rest,
    sample_number,
    save_data,
    nb_states,
    nb_states_scale,
    quantize_read,
    quantize_write,
    noise_input,
    noise_bias,
    input_scale,
    pos_mult,
    neg_mult,
    var_dw,
    var_vprog,
    var_w,
    fraction_non_zero,
):
    # weights_before = weights.copy()
    # weights ->  weights[input neuron, spiking_neuron] 
    sample_dW_pot = np.zeros((mem_pot_input.shape[0], 1))
    sample_dW_dep = np.zeros((mem_pot_input.shape[0], 1))
    # sample_dW_pot = []
    # sample_dW_dep = []

    refractory_neurons_input = np.zeros(mem_pot_input.shape[0])
    refractory_neurons = np.zeros(mem_pot_output.shape[0])
    recorded_input_spikes = np.zeros((mem_pot_input.shape[0], duration_per_sample))
    recorded_output_spikes = np.zeros((mem_pot_output.shape[0], duration_per_sample))

    for t in range(duration_per_sample): 
        thresholds = (thresholds - threshold_rest) * th_leak_cst + threshold_rest
        refractory_neurons = np.maximum(0.0, refractory_neurons - 1)
        refractory_neurons_input = np.maximum(0.0, refractory_neurons_input - 1)
        non_refrac_neurons = refractory_neurons == 0
        non_refrac_input_neurons = refractory_neurons_input == 0


        input_leak_cst_positive = input_leak_cst 

        mem_pot_input[non_refrac_input_neurons] = (
            np.where(mem_pot_input[non_refrac_input_neurons] > 0, 
                    mem_pot_input[non_refrac_input_neurons] * input_leak_cst_positive,
                    mem_pot_input[non_refrac_input_neurons] * input_leak_cst_negative)
        ) #Implement leak

        mem_pot_input[non_refrac_input_neurons]+= (
            np.where(X[non_refrac_input_neurons] > 0.1*input_scale, 
                    X[non_refrac_input_neurons] * (1+np.random.normal(0, noise_input * input_scale, X[non_refrac_input_neurons].shape)),
                    input_bias * (1+np.random.normal(0, noise_bias*input_bias, X[non_refrac_input_neurons].shape)))
        ) #Add input bias and noise

        # mem_pot_input[non_refrac_input_neurons] += np.random.normal(0, noise_input*input_bias, X[non_refrac_input_neurons].shape)
        
        mem_pot_input[~non_refrac_input_neurons] = input_reset  # Refractory neurons are fixed at -1
        input_spikes = mem_pot_input > input_threshold
        mem_pot_input[input_spikes] = input_reset  # reset
        recorded_input_spikes[input_spikes, t] = 1
        refractory_neurons_input[input_spikes] = refractory_period_input


        if quantize_read:
            if(nb_states_scale == 'linear'):
                weights_quantized = quantize_linear(weights, nb_states)
                weights_effective = (wmax-wmin) * weights_quantized + wmin
            elif(nb_states_scale == 'log'):
                weights_quantized = quantize_log(weights, nb_states)
                weights_effective = weights_quantized * (wmax - wmin) + wmin # Scale back to [wmin,wmax]          
        else:
            weights_effective = weights * (wmax - wmin) + wmin
          
        
        mem_pot_output[non_refrac_neurons] = mem_pot_output[non_refrac_neurons] * output_leak_cst + input_spikes.astype(
            np.float64
        ) @ (
            weights_effective[:, non_refrac_neurons]
        )  # wmin is LRS/HRS
        output_spikes = mem_pot_output > thresholds
        if np.any(output_spikes):
            spiking_neuron = np.argmax(mem_pot_output - thresholds)  # This neuron is spiking
            thresholds[spiking_neuron] += th_inc
            if use_vdsp:

                vapplied = mem_pot_input - vprog
                if( np.isscalar(vdsp_lr)==0 ):
                    vdsp_lr_spiking=vdsp_lr[:, spiking_neuron]
                else:
                    vdsp_lr_spiking=vdsp_lr

                if( np.isscalar(alpha)==0):
                    alpha_spiking=alpha[:, spiking_neuron]
                else:
                    alpha_spiking=alpha

                if( np.isscalar(gamma)==0):
                    gamma_spiking=gamma[:, spiking_neuron]
                else:
                    gamma_spiking=gamma
                
                if( np.isscalar(lr_pn)==0):
                    lr_pn_spiking=lr_pn[:, spiking_neuron]
                else:
                    lr_pn_spiking=lr_pn

                if( np.isscalar(alpha_pn)==0):
                    alpha_pn_spiking=alpha_pn[:, spiking_neuron]
                else:
                    alpha_pn_spiking=alpha_pn

                if( np.isscalar(gamma_pn)==0):
                    gamma_pn_spiking=gamma_pn[:, spiking_neuron]
                else:
                    gamma_pn_spiking=gamma_pn

                if( np.isscalar(vth)==0):
                    vth_spiking=vth[:, spiking_neuron]
                else:
                    vth_spiking=vth

                if( np.isscalar(vth_pn)==0):
                    vth_pn_spiking=vth_pn[:, spiking_neuron]
                else:
                    vth_pn_spiking=vth_pn    


                # vapplied = mem_pot_input-vprog
                vapplied = mem_pot_input

                if var_vprog > 0:
                    vapplied = vapplied*np.random.uniform(1-var_vprog, 1+var_vprog, size=vapplied.shape)

                if var_w > 0:
                    shape = weights[:, spiking_neuron].shape
                    weights_with_var = weights[:, spiking_neuron] * np.random.uniform(1-var_w, 1+var_w, size=shape)
                    weights_with_var = clip(weights_with_var, 0, 1)
                else:
                    weights_with_var = weights[:, spiking_neuron]


                vapplied_scaled = (vapplied * pos_mult * (vapplied > 0)) + (vapplied * neg_mult * (vapplied < 0))
                # dw = vdsp_memristor(weights[:, spiking_neuron], vapplied_scaled, vdsp_lr_spiking,lr_pn_spiking,gamma_spiking,gamma_pn_spiking,alpha_spiking,alpha_pn_spiking,vth_spiking,vth_pn_spiking)
                dw = vdsp_memristor(weights_with_var, vapplied_scaled, vdsp_lr_spiking,lr_pn_spiking,gamma_spiking,gamma_pn_spiking,alpha_spiking,alpha_pn_spiking,vth_spiking,vth_pn_spiking)

                fraction_non_zero += np.count_nonzero(dw) / dw.size

                if var_dw > 0:
                    #Sample an array of random numbers between 1-var_dw and 1+var_dw and multiply with dw
                    dw = dw * np.random.uniform(1-var_dw, 1+var_dw, size=dw.shape)
                #Stochastic switching
                weights_to_update = np.random.uniform(0, 1, size=dw.shape) < switching_prob
                
                weights[:, spiking_neuron] += dw * weights_to_update  # Apply plasticity
               
                #Stuck devices
                weights = weights * HRS_mask
                weights = weights + LRS_mask  # LRS: w=1
                weights = clip(weights, 0, 1)

                # Take difference bwtween 
                # dw = weights[:, spiking_neuron] - weights_initial[:, spiking_neuron]

                # if save_data:
                #     sample_dW_pot = sample_dW_pot.flatten() + (dw.flatten() * (dw.flatten() > 0))
                #     sample_dW_dep = sample_dW_dep.flatten() +  (dw.flatten() * (dw.flatten() < 0))


            mem_pot_output[spiking_neuron] = 0  # reset mem pot
            refractory_neurons[spiking_neuron] = refractory_period  # Set refrac
            recorded_output_spikes[spiking_neuron, t] = 1

            # For all other output neurons, do lateral inhibition (clamp at 0 for lateral_inhibition_period)
            non_spiking_neurons = np.ones(mem_pot_output.shape[0], dtype=np.bool_)
            non_spiking_neurons[spiking_neuron] = False
            mem_pot_output[non_spiking_neurons] = 0
            refractory_neurons[non_spiking_neurons] = lateral_inhibition_period

    mem_pot_input = mem_pot_input * np.power(input_leak_cst, duration_between_samples)
    mem_pot_output = mem_pot_output * np.power(output_leak_cst, duration_between_samples)
    refractory_neurons = np.maximum(0.0, refractory_neurons - duration_between_samples)

            #     if 'dW_pot_history_updated' in locals():
            #     pass
            # else:
            #     dW_history_updated = dW_history
            #     sample_number_updated = sample_history
    
    if save_data:
            dW_pot_history[:, sample_number] = sample_dW_pot.flatten()
            dW_dep_history[:, sample_number] = sample_dW_dep.flatten()
    #     # print("shape of dW_pot_history", dW_pot_history.shape)
    #     # print("shape of sample_dW_pot", sample_dW_pot.shape)
    #     # dW_pot_history_updated = np.append(dW_pot_history, sample_dW_pot)
    #     # dW_dep_history_updated = np.append(dW_dep_history, sample_dW_dep)

    # #     sample_number_array = np.ones_like(sample_dW_pot) * sample_number
    # #     sample_number_updated = sample_number_array
    # # else:
    #     dW_pot_history_updated = dW_pot_history
    #     dW_dep_history_updated = dW_dep_history
    #     sample_number_updated = sample_history

    return mem_pot_input, mem_pot_output, weights,weights_effective, recorded_input_spikes, recorded_output_spikes, dW_pot_history, dW_dep_history ,sample_number, fraction_non_zero

def save_dict_to_file(dic, filename):
    f = open(filename, "w")
    f.write(str(dic))
    f.close()


def compute_input_bias(input_leak_cst, v_th, input_bias_scale=0.9):
    return -v_th * np.log(input_leak_cst) * input_bias_scale


def main(
    seed=0x1B,
    n_output_neurons=50,
    duration_per_sample=40,
    duration_between_samples=100,
    input_leak_cst=np.exp(-0.001 / 0.030),
    input_leak_cst_negative=np.exp(-0.001 / 0.030),
    output_leak_cst=np.exp(-0.001 / 0.060),
    input_threshold=1.8,
    # input_threshold_max = 2,
    output_threshold=10,
    input_reset=-1.8,
    # input_reset_min=-2,
    input_bias_scale=0.98,
    refractory_period=5,
    lateral_inhibition_period=50,
    refractory_period_input=10,
    input_scale=0.01,
    nb_epochs=1,
    use_vdsp=True,
    device='tio2',
    vdsp_lr=0.1,
    lr_pn=1,
    gamma=1,
    gamma_pn=1,
    alpha=1,
    alpha_pn=1,
    vth=0.1,
    vth_pn=1,
    wmin=0.1,  # LRS/HRS
    wmax=1,
    vprog=0,
    switching_prob=1,
    stuck_LRS=0,
    stuck_HRS=0,
    read_variability=0,
    write_variability=0,
    clipen=0,
    vclipmin=-2,
    vclipmax=2,
    weight_init_scale_min=0,
    weight_init_scale_max=1,
    with_validation=False,
    th_leak_cst=np.exp(-1 / 1000),
    th_inc=0.0,  # 0 is no adaptive threshold
    with_plots=False,
    normalize_duration=False,
    save_data=False,
    nb_states = 4,
    quantize_write = False,
    quantize_read = False,
    nb_states_scale = 'linear',
    noise_input = 0,
    noise_bias = 0,
    train_size = 60000,
    test_size = 10000,
    pos_mult_init = 1,
    neg_mult_init = 1,
    pos_mult_step = 0.1,
    neg_mult_step = 0.1,
    pos_mult_step_decay = 0.9,
    neg_mult_step_decay = 0.9,
    pos_mult_max = 1,
    neg_mult_max = 1,
    nbsteps = 10, # Total number of batches for increasing pos_mult and neg_mult
    supress_output = False,
    presentation_time_multiplier = 1,
    var_dw = 0,
    var_vprog = 0,
    var_w = 0
):
    # if(device=='hzo'):
    #     memristor_params = get_hzo_params()
    # else:
    #     memristor_params = get_tio2_params()

    input_bias = compute_input_bias(input_leak_cst, input_threshold, input_bias_scale)
    # memristor_params = get_hzo_params()
    #If not suppressing output, print the parameters
    if not supress_output:
        print(locals())
    argument_dict = locals()
    filename = f"accuracy_{seed}.csv"
    # save_dict_to_file(argument_dict, filename)

    df = pd.DataFrame.from_dict(argument_dict, orient="index")
    df = df.transpose()
    df.to_csv(filename, index=True)

    np.random.seed(seed)
    mndata = MNIST()
    images, labels = mndata.load_training()
    X_train, y_train = np.asarray(images), np.asarray(labels)
    X_train = X_train / 255 * input_scale


    images, labels = mndata.load_testing()
    X_test, y_test = np.asarray(images), np.asarray(labels)
    X_test = X_test / 255 * input_scale
    
    
    if train_size < 10000:
        X_validation, y_validation = X_train[10000:20000], y_train[10000:20000]
    elif train_size < 60000:
        #Take the first 10000 of the train size 
        X_validation, y_validation = X_train[0:10000], y_train[0:10000]
    elif train_size == 60000:
        X_validation, y_validation = X_train, y_train
    else:
        print("Error: train_size should be less than 60000")
    X_train = X_train[:train_size] #Take the first train_size samples
    y_train = y_train[:train_size]


    X_test = X_test[:test_size]
    y_test = y_test[:test_size]
    
    mem_pot_input = np.zeros(784)
    mem_pot_output = np.zeros(n_output_neurons)
    weights = np.random.uniform(weight_init_scale_min, weight_init_scale_max, size=(784, n_output_neurons))
    initial_weights = weights.copy()


    fraction_non_zero = 0 # Fraction of non-zero weight update events


    LRS_mask = np.random.uniform(0, 1, size=weights.shape) < stuck_LRS  # obtain 1 for stuck devices
    HRS_mask = np.random.uniform(0, 1, size=weights.shape) > stuck_HRS  # obtain 0 for stuck devices

    thresholds = np.ones(n_output_neurons) * output_threshold
    max_freq = 0
    max_output_freq = 0
    nbsamples = X_train.shape[0] * nb_epochs

    dW_pot_history = np.zeros((784, nbsamples))
    dW_dep_history = np.zeros((784, nbsamples))
    # dW_dep_history = np.zeros((nbsamples, 784))


    # dW_dep_history_updated = []
    sample_number_updated = []
    sample_number = 0

    #Data saving for plotting
    # recorded_input_spikes_all = np.zeros((nbsamples, mem_pot_input.shape[0], duration_per_sample))
    # recorded_output_spikes_all = np.zeros((nbsamples, mem_pot_output.shape[0], duration_per_sample))
    nb_spikes_out_total = 0

    step_count = 0  # Initialize step count for exponential increase in step size

    spike_counts = np.zeros((10, n_output_neurons))  # For every class, keep track of spike count per neuron

    step_size_samples = train_size // nbsteps  # Number of samples before increasing threshold and reset : batch size
    # Initialize lists to store values for plotting
    # Loop with exponentially increasing step size
    pos_mult = pos_mult_init
    neg_mult = neg_mult_init

    for epoch in range(nb_epochs):
        # if not supress_output:
        pbar = tqdm(X_train, disable=supress_output)
        for i, (X, y) in enumerate(zip(pbar, y_train)):

            # Uncomment to save weights every 15000 samples
            # if save_data:
            #     if (i % 120000 == 0):
            #         dW_history_updated = []
            #         sample_number_updated = []
            # print(i)
            sample_number += 1

            # input_threshold_init = input_threshold
            # input_reset_init = input_reset
            # input_threshold = input_threshold_max - (input_threshold_max - input_threshold) * i / train_size
            # input_reset = input_reset + (input_reset_min - input_reset) * i / train_size

            # for i in range(train_size):
            # if i % step_size_samples == 0 and i != 0:
            #     step_size_pos_mult_exponential = pos_mult_step #* (pos_mult_step_decay ** step_count)
            #     step_size_neg_mult_exponential = neg_mult_step #* (neg_mult_step_decay ** step_count)
            #     pos_mult += step_size_pos_mult_exponential
            #     neg_mult += step_size_neg_mult_exponential
            #     # Clip threshold value
            #     # print(pos_mult, neg_mult)
            #     pos_mult = min(pos_mult_max, pos_mult)
            #     # Clip reset value
            #     neg_mult = min(neg_mult_max, neg_mult)
            #     step_count += 1
                # print(i, pos_mult, neg_mult, step_count)
            
            # if sample_number>0:

            #     dW_dep_history = dW_dep_history_updated
            #     dW_pot_history = dW_pot_history_updated
            
            mem_pot_input, mem_pot_output, weights, weights_effective, recorded_input_spikes, recorded_output_spikes, dW_pot_history, dW_dep_history, sample_number_updated, fraction_non_zero = run_one_sample(
                X,
                mem_pot_input,
                mem_pot_output,
                weights,
                dW_pot_history,
                dW_dep_history,
                sample_number_updated,
                duration_per_sample,
                duration_between_samples,
                input_leak_cst,
                input_leak_cst_negative,
                input_bias,
                input_threshold,
                input_reset,
                output_leak_cst,
                refractory_period,
                refractory_period_input,
                lateral_inhibition_period,
                use_vdsp=use_vdsp,
                vdsp_lr=vdsp_lr,
                lr_pn=lr_pn,
                gamma=gamma,
                gamma_pn=gamma_pn,
                alpha=alpha,
                alpha_pn=alpha_pn,
                vth=vth,
                vth_pn=vth_pn,
                wmin=wmin,
                wmax=wmax,
                vprog=vprog,
                switching_prob=switching_prob,
                LRS_mask=LRS_mask,
                HRS_mask=HRS_mask,
                read_variability=read_variability,
                write_variability=write_variability,
                clipen=clipen,
                vclipmin=vclipmin,
                vclipmax=vclipmax,
                thresholds=thresholds,
                th_inc=th_inc,
                th_leak_cst=th_leak_cst,
                threshold_rest=output_threshold,
                sample_number=sample_number,
                save_data=save_data,
                nb_states= nb_states,
                nb_states_scale = nb_states_scale,
                quantize_read=quantize_read,
                quantize_write=quantize_write,
                noise_input=noise_input,
                noise_bias=noise_bias,
                input_scale=input_scale,
                pos_mult=pos_mult,
                neg_mult=neg_mult,
                var_dw=var_dw,
                var_vprog=var_vprog,
                var_w=var_w,
                fraction_non_zero=fraction_non_zero
            )

            input_freqs = recorded_input_spikes.mean(axis=1) * 1000
            output_freqs = recorded_output_spikes.sum(axis=1) * 1000
            max_freq = np.maximum(max_freq, np.max(input_freqs))
            max_output_freq = np.maximum(max_output_freq, np.max(output_freqs))
            # if normalize_duration:
            #     if max_freq > 0:
            #         duration_per_sample = int(presentation_time_multiplier * 1000 / max_freq)
            #         duration_per_sample = 40
            #     else:
            #         duration_per_sample = 40
                # duration_per_sample = int(1000/max_freq)
            #Calculate the minimum frequency, number of non-zero elements and the number of zero elements, for the input and output neurons 
            # min_freq = np.min(input_freqs)
            # input_freqs_non_zero = np.count_nonzero(input_freqs)
            # input_freqs_zero = len(input_freqs) - input_freqs_non_zero
            # output_freqs_non_zero = np.count_nonzero(output_freqs)
            # output_freqs_zero = len(output_freqs) - output_freqs_non_zero
            # #Calculate min and max frequency for the output neurons
            # min_output_freq = np.min(output_freqs)
            #Set the progress bar postfix with min amd max frequency for the input and output neurons
            nb_spikes_out_total += np.sum(recorded_output_spikes)
            if not supress_output:
                pbar.set_postfix({"nb_spikes_out_total": nb_spikes_out_total, "Duration_per_sample": duration_per_sample})
            
            # pbar.set_postfix({"max_input_freq": max_freq, "0_freq": input_freqs[0], "duration_per_sample": duration_per_sample})
            sample_number_to_save = [100,300, 600, 1000,2000, 3000, 5000, 6000,10000, 12000,20000, 24000, 30000, 40000 , 50000, 60000]
            if  (i + 1) in sample_number_to_save:
                if with_plots:
            # if with_plots and (i + 1) % 15000 == 0:
                    fig, axs = plt.subplots(5, n_output_neurons // 5, figsize=((3/2)*(n_output_neurons//5), 7))
                    plt.rc('font', size=14)

                    # fig.suptitle(f"Presynaptic weights upto sample number {i+1}")
                    axs = axs.flatten()
                    for neuron in range(n_output_neurons):
                        # if i > len(y_train) // 2:
                        #    axs[neuron].set_xlabel(f"{np.argmax(spike_counts, axis=0)[neuron]}")
                        #    axs[neuron].get_yaxis().set_visible(False)
                        #    axs[neuron].set_xticks([])
                        # else:
                        axs[neuron].set_axis_off()
                        axs[neuron].imshow(weights[:, neuron].reshape(28, 28) )# ,vmin=0,vmax=1)  # , cmap="hot"
                        axs[neuron].set_title(f"O{neuron+1}")
                        # axs.set
                    # Reduce white space between subplots
                    plt.subplots_adjust(wspace=0, hspace=0)
                    plt.tight_layout()
                    fig.savefig(f"plots_local/{seed}_mnist_wta_epoch_{epoch:02}_iteration_{i+2:05}.jpg", bbox_inches="tight", dpi=300)
                    fig.savefig(f"plots_local/high_res/{seed}_mnist_wta_epoch_{epoch:02}_iteration_{i+2:05}.svg", bbox_inches="tight", dpi=300)
                    plt.close(fig)
                    fig,axs = plt.subplots(1,1,figsize=(6,4))
                    plt.hist(weights.flatten(), density=True, facecolor='g', alpha=0.75)
                    plt.xlim(0, 1)
                    plt.ylabel("pdf")
                    plt.xlabel("W")
                    plt.ylabel("Density")
                    #Set font size to 14
                    plt.rc('font', size=14)

                    plt.tight_layout()
                    plt.savefig(f"plots_local/{seed}_histogram_mnist_wta_epoch_{epoch:02}_iteration_{i+2:05}.jpg", bbox_inches="tight", dpi=300)
                    plt.savefig(f"plots_local/high_res/{seed}_histogram_mnist_wta_epoch_{epoch:02}_iteration_{i+2:05}.svg", bbox_inches="tight", dpi=300)
                    

                    plt.close(fig)
    print("Training done")
    print("fraction_non_zero", fraction_non_zero)    
    # Redo training set to associate labels
    spike_counts = np.zeros((10, n_output_neurons))  # For every class, keep track of spike count per neuron
    pbar = tqdm(X_validation, disable=supress_output)
    for i, (X, y) in enumerate(zip(pbar, y_validation)):
        mem_pot_input, mem_pot_output, weights, weights_effective ,recorded_input_spikes, recorded_output_spikes, dW_pot_history, dW_dep_history ,sample_number_updated, fraction_non_zero = run_one_sample(
            X,
            mem_pot_input,
            mem_pot_output,
            weights,
            dW_pot_history,
            dW_dep_history,
            sample_number,
            duration_per_sample,
            duration_between_samples,
            input_leak_cst,
            input_leak_cst_negative,
            input_bias,
            input_threshold,
            input_reset,
            output_leak_cst,
            refractory_period,
            refractory_period_input,
            lateral_inhibition_period=0,
            use_vdsp=False, #Disable learning for assigning labels
            vdsp_lr=0.0,#Disable learning for assigning labels
            lr_pn=lr_pn,
            gamma=gamma,
            gamma_pn=gamma_pn,
            alpha=alpha,
            alpha_pn=alpha_pn,
            vth=vth,
            vth_pn=vth_pn,
            wmin=wmin,
            wmax=wmax,
            vprog=vprog,
            switching_prob=switching_prob,
            LRS_mask=LRS_mask,
            HRS_mask=HRS_mask,
            read_variability=read_variability,
            write_variability=write_variability,
            clipen=clipen,
            vclipmin=vclipmin,
            vclipmax=vclipmax,
            thresholds=thresholds,
            th_inc=0,   # Keep threshold as is
            th_leak_cst=th_leak_cst,
            threshold_rest=output_threshold,
            sample_number=i,
            save_data=False,
            nb_states=-1, # -1 means no discretization
            nb_states_scale=nb_states_scale,
            quantize_read=False,
            quantize_write=False,
            noise_input=0, # No noise for assigning labels
            noise_bias=0,
            input_scale=input_scale,
            pos_mult=1, # No need to change the weights
            neg_mult=1,
            var_dw=0,
            var_vprog=0,
            var_w=0,
            fraction_non_zero=fraction_non_zero
        )
        spike_counts[y] += np.sum(recorded_output_spikes, axis=1)

    # Normalize the spike counts per neuron
    # sum_spike_counts = spike_counts.sum(axis=1)

    # Associate a label for every neuron based on its highest spike count per class
    labels = np.argmax(spike_counts, axis=0)
    nb_correct_classification_method_1 = 0
    nb_correct_classification_method_2 = 0
    nb_correct_classification_method_3 = 0

    # recorded_input_spikes_all = np.zeros((X_test.shape[0], mem_pot_input.shape[0], duration_per_sample))
    recorded_output_spikes_all = np.zeros((X_test.shape[0], mem_pot_output.shape[0], duration_per_sample))

    #plot the weights but with labels associated with them
    if with_plots:
            fig, axs = plt.subplots(5, n_output_neurons // 5, figsize=((3/2)*(n_output_neurons//5), 7))
            plt.rc('font', size=14)

            # fig.suptitle(f"Presynaptic weights upto sample number {i+1}")
            axs = axs.flatten()
            for neuron in range(n_output_neurons):

                axs[neuron].set_axis_off()

                axs[neuron].imshow(weights[:, neuron].reshape(28, 28) )# ,vmin=0,vmax=1)  # , cmap="hot"
                
                # axs[neuron].set_title(f"O{neuron+1}")
                axs[neuron].set_title(f"{np.argmax(spike_counts, axis=0)[neuron]}")
                # axs.set
            # Reduce white space between subplots
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout()
            fig.savefig(f"plots_local/{seed}final.jpg", bbox_inches="tight", dpi=300)
            plt.close(fig)


    for i, (X, y) in enumerate(zip(tqdm(X_test, disable=supress_output), y_test)):
        mem_pot_input, mem_pot_output, weights, weights_effective,recorded_input_spikes, recorded_output_spikes, dW_pot_history, dW_dep_history ,sample_number_updated, fraction_non_zero = run_one_sample(
            X,
            mem_pot_input,
            mem_pot_output,
            weights,
            dW_pot_history,
            dW_dep_history,
            sample_number_updated,
            duration_per_sample,
            duration_between_samples,
            input_leak_cst,
            input_leak_cst_negative,
            input_bias,
            input_threshold,
            input_reset,
            output_leak_cst,
            refractory_period,
            refractory_period_input,
            lateral_inhibition_period,
            use_vdsp=False,
            vdsp_lr=0,
            lr_pn=lr_pn,
            gamma=gamma,
            gamma_pn=gamma_pn,
            alpha=alpha,
            alpha_pn=alpha_pn,
            vth=vth,
            vth_pn=vth_pn,
            wmin=wmin,
            wmax=wmax,
            vprog=vprog,
            switching_prob=switching_prob,
            LRS_mask=LRS_mask,
            HRS_mask=HRS_mask,
            read_variability=read_variability,
            write_variability=write_variability,
            clipen=clipen,
            vclipmin=vclipmin,
            vclipmax=vclipmax,
            thresholds=thresholds,
            th_inc=0, # Keep threshold as is
            th_leak_cst=th_leak_cst,  # Keep threshold as is
            threshold_rest=output_threshold,
            sample_number=i,
            save_data=save_data,
            nb_states=-1, # Weights already quantized
            nb_states_scale=nb_states_scale,
            quantize_read=False,
            quantize_write=False,
            noise_input=0, # No noise for testing
            noise_bias=0,
            input_scale=input_scale,
            pos_mult=1, # No need to change the weights
            neg_mult=1,
            var_dw=0,
            var_vprog=0,
            var_w=0,
            fraction_non_zero=fraction_non_zero
        )

        # recorded_input_spikes_all[i,:,:] = recorded_input_spikes
        recorded_output_spikes_all[i,:,:] = recorded_output_spikes
        ## Method 1: find the top spiking neuron and get its label
        sum_of_output_spikes = np.sum(recorded_output_spikes, axis=1)  # / sum_spike_counts
        top_spiking_neuron = np.argmax(sum_of_output_spikes)
        output_label = np.argmax(spike_counts[:, top_spiking_neuron])
        nb_correct_classification_method_1 += y == output_label

        ## Method 2: sum the spikes for all labeled neurons
        sum_of_spikes_for_sample = np.zeros(10)
        np.add.at(sum_of_spikes_for_sample, labels, sum_of_output_spikes)
        output_label = np.argmax(sum_of_spikes_for_sample)
        nb_correct_classification_method_2 += y == output_label

        ## Method 3: sum the spikes for all labeled neurons and divide by the number of neurons in that class
        sum_of_spikes_for_sample = np.zeros(10)
        np.add.at(sum_of_spikes_for_sample, labels, sum_of_output_spikes)
        sum_of_spikes_for_sample = sum_of_spikes_for_sample / np.bincount(labels)[output_label]
        output_label = np.argmax(sum_of_spikes_for_sample)
        nb_correct_classification_method_3 += y == output_label        


    if not supress_output:
        print(f"Accuracy of method 1: {nb_correct_classification_method_1 / len(y_test):.5f}")
        print(f"Accuracy of method 2: {nb_correct_classification_method_2 / len(y_test):.5f}")
        print(f"Accuracy of method 3: {nb_correct_classification_method_3 / len(y_test):.5f}")
        print(f"Maximum spiking frequency of input layer: {max_freq}")
    max_accuracy=max(nb_correct_classification_method_1 / len(y_test), nb_correct_classification_method_2 / len(y_test), nb_correct_classification_method_3 / len(y_test))

    # freq_return = np.max(max_freq,0)
    # print(np.array([max_accuracy, freq_return]))
    nb_spikes_out_per_sample = nb_spikes_out_total/train_size
    answer = np.array([max_accuracy, max_freq, nb_spikes_out_per_sample, duration_per_sample, fraction_non_zero], dtype=np.float64)
    print(answer.shape)
    print(answer)
    # np.savez(f"{seed}_test_spikes.npz",  recorded_input_spikes_all=recorded_input_spikes_all, recorded_output_spikes_all=recorded_output_spikes_all)
    
    return answer

def run_vdsp(
    output_threshold=10,
    **args,
):
    return main(
        output_threshold=output_threshold,
        **args,
    )

def run_vdsp_vs_freq(seed, iteration, n_samples, normalize_duration=True):
    input_scale = np.logspace(-3.05, 0, n_samples)[iteration]
    acc, max_freq = run_vdsp(seed=seed, input_scale=input_scale, normalize_duration=normalize_duration)
    with open(f"vdsp_vs_freq_{iteration}_of_{n_samples}_seed_{seed}.txt", "w") as f:
        f.write(f"{acc}\n{max_freq}\n")

if __name__ == "__main__":
    Fire()