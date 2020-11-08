import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt

from visbrain.objects import TopoObj, SceneObj


def plot_ir_index_hist(config):
    # Read the data
    read_path = config['processed_data']['eeg_epochs_ir_path']
    data = dd.io.load(read_path)
    ir_index = []
    for subject in config['subjects']:
        ir_index.append(data[subject]['targets'])

    ir_index = np.concatenate(ir_index, axis=0)
    plt.style.use('clean')
    plt.hist(ir_index, bins=100)
    plt.show()
    return None


def plot_selected_sensors(features):
    sc = SceneObj(bgcolor='white', size=(600, 600))
    # Define some EEG channels and set one data value per channel
    ch_names = [
        'Fp1', 'F7', 'F8', 'T4', 'T6', 'T5', 'T3', 'Fp2', 'O1', 'P3', 'Pz',
        'F3', 'Fz', 'F4', 'C4', 'P4', 'POz', 'C3', 'Cz', 'O2'
    ]
    selected_names = [name.split('_')[0] for name in features]

    # Create the topoplot and the associated colorbar
    kw_top = dict(margin=15 / 100, chan_offset=(0., -0.045, 0.), chan_size=10)
    t_obj_selected = TopoObj('topo', [1] * len(selected_names),
                             channels=selected_names,
                             chan_mark_color='#F19D99',
                             cmap='Greys',
                             verbose=False,
                             **kw_top)
    sc.add_to_subplot(t_obj_selected,
                      row=0,
                      col=0,
                      title='Selected Channels',
                      title_color='black',
                      width_max=600)
    t_obj = TopoObj('topo', [1] * len(ch_names),
                    channels=ch_names,
                    chan_mark_color='#F19D99',
                    verbose=False,
                    **kw_top)
    # Add both objects to the scene
    sc.add_to_subplot(t_obj, row=0, col=0, width_max=600)
    sc.preview()
