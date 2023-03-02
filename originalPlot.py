import pickle
from scipy.stats import sem ,t
from scipy import mean
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import json


color = {
    'qmix': [176, 58, 46],
    'vdn': [125, 60, 152],
    'nmaddpg': [46, 134, 193],
    'maddpg': [19, 141, 117],
    'qmix_maddpg': [40, 180, 99],
    'vdn_maddpg': [214, 137, 16],
}

color = {key: np.array(value, np.float) / 255. for key, value in color.items()}

constructions = [
    'vdn',
    'qmix',
    'nmaddpg',
    'maddpg',
    'qmix_maddpg',
    'vdn_maddpg',
]

policy_grad_constructions = [
    'nmaddpg',
    'maddpg',
    'qmix_maddpg',
    'vdn_maddpg',
    'vdn',
]

construction_labels = [
    'VDN',
    'QMIX',
    'NMADDPG',
    'MADDPG',
    'Q-MADDPG',
    'V-MADDPG',
]

algorithm_lines = [
    '-',
    '-',
    '-',
    '-',
    '-',
    '-',
    '-',
    '-',
    '--',
]

#envs = ['Pred-prey miscap0']

envs = [
    'Pred-prey miscap0 mixed',
    'Matrix Game Set 2',
]

# #################

alpha = 0.2
scale = 50.
confidence = 0.95
log_scale = False
font_size = 26
legend_font_size = 32
anchor = (0.5, 1.08)


def smooth(data, alg):
    start = 0
    range1 = 5
    if alg == 'dicg':
        return data
    new_data = np.zeros_like(data)
    for i in range(int(start), int(range1)):
        new_data[i] = 1. * sum(data[0 : i + 1]) / (i + 1)
    for i in range(int(range1), len(data)):
        new_data[i] = 1. * sum(data[i - int(range1) + 1 : i + 1]) / range1

    return new_data
    #return data;


def resize(data):
    if len(data) < max_length:
        data += [0 for _ in range(max_length - len(data))]
    elif len(data) > max_length:
        data = data[:max_length]

    return data


def read_data(construction, loss, cut_length=None):
    data_n = []
    x_n = []

    files = []
    path = 'Result Sets/' + env + '/' + construction
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith('info.json') or construction == 'dicg' and file.endswith('.npy'):
                files.append(os.path.join(r, file))

    run_number = len(files)

    for f in files:
        with open(f, 'r') as _f:
            print(f)

            d = json.load(_f)
            data = np.array([])

            # To get test return means
            if(policy_grad_constructions.__contains__(construction)):
                for i in d['test_return_mean']:
                    data = np.append(data, i['value'])
                data_n.append(data)
                x_n.append(np.array(d["test_return_mean_T"]))
            else:
                for i in d['test_return_mean']:
                    data = np.append(data, i['value'])
                data_n.append(data)
                x_n.append(np.array(d["test_return_mean_T"]))
            
            '''
            # X and Y Values
            data_n.append(np.array(d['q_taken_mean']))
            x_n.append(np.array(d['q_taken_mean_T']))
            '''
            

    min_length = min([len(_x) for _x in x_n] + [cut_length])
    data_n = [data[:min_length] for data in data_n]
    x_n = [x[:min_length] for x in x_n]

    print(data_n)
    print(x_n)
    if(env == "Matrix Game Set 2"):
        return np.array(x_n), 100*np.array(data_n), min_length, run_number
    else:
        return np.array(x_n), np.array(data_n), min_length, run_number


def smooth_tderror(data):
    start = 0
    range1 = 4.0
    new_data = np.zeros_like(data)
    for i in range(int(start), int(range1)):
        new_data[i] = 1. * sum(data[0 : i + 1]) / (i + 1)
    for i in range(int(range1), len(data)):
        new_data[i] = 1. * sum(data[i - int(range1) + 1 : i + 1]) / range1

    return new_data


def read_data_tderror(construction, loss, cut_length=None):
    data_n = []
    x_n = []

    files = []
    path = 'Result Sets/'+env + '/' + construction
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith('info.json'):
                files.append(os.path.join(r, file))

    run_number = len(files)

    for f in files:
        with open(f, 'r') as _f:
            print(f)
            d = json.load(_f)

            data_n.append(np.array(d['td_error_abs']))
            x_n.append(np.array(d['td_error_abs_T']))

    min_length = min([len(_x) for _x in x_n] + [cut_length])
    
    data_n = [data[:min_length] for data in data_n]
    x_n = [x[:min_length] for x in x_n]

    return np.array(x_n), np.array(data_n), min_length, run_number

# Main function here:
s_cut = 200

if __name__ == '__main__':
    # figure =
    # ######### 10
    figure = None
    figure = plt.figure(figsize=(66, 13))

    data = [[] for _ in constructions]

    legend_elements = [Line2D([0], [0], lw=4, label=label, color=color[construction], linestyle=style) for (construction, label, style) in
                       zip(constructions, construction_labels, algorithm_lines)]
    figure.legend(handles=legend_elements, loc='upper center', prop={'size': legend_font_size}, ncol=min(len(constructions), 5),
                  bbox_to_anchor=(0.5, 1.25), frameon=False)

    # Create plot for each environment's loss (so graph setup for loss (previously win rate))
    for idx, env in enumerate(envs):
        ax = None
        ax= plt.subplot(1, 3, idx+1)

        ax.grid()
        # figure = plt.figure()
        # plt.grid()
        method_index = 0

        for (construction, style) in zip(constructions, algorithm_lines):
            x, y, min_length, run_number = read_data(construction, env, cut_length=501)
            print(env, construction, run_number)

            if run_number == 0:
                continue

            y_mean = smooth(np.mean(y, axis=0), construction) #y_mean = smooth(np.mean(y, axis=0))
            train_scores_mean = y_mean
            data[method_index].append(y_mean[:s_cut])
            method_index += 1

            low = smooth(np.percentile(y, 25, axis=0), construction)
            high = smooth(np.percentile(y, 75, axis=0), construction)
        
            #h = smooth(sem(y) * t.ppf((1 + confidence) / 2, min_length - 1))
            #    h = smooth(sem(data) * t.ppf((1 + confidence) / 2, max_length - 1))
            bhos = np.linspace(1, min_length, min_length)
            bhos = x[0] / 1000000
            # if log_scale:
            #     train_scores_mean = np.log(train_scores_mean + scale) - np.log(scale)
            #     h = np.log(h + scale) - np.log(scale)
            ax.fill_between(bhos, low,
                             high, alpha=alpha,
                             color=color[construction], linewidth=0)
            width = 4
            ax.plot(bhos, train_scores_mean, color=color[construction], label=construction, linewidth=width, linestyle=style)

        # Others
        if idx == 0:
            title = 'a) pred-prey Mean Episode Return'
        else:
            title = 'b) Matrix Game Mean Episode Return'
        ax.tick_params('x', labelsize=font_size)
        ax.tick_params('y', labelsize=font_size)
        ax.set_xlabel('T (mil)', size=font_size)
        ax.set_ylabel('Mean', size=font_size)
        ax.set_title(title, size=legend_font_size)
        ax.set_ylim(-5, 105)

    # Starting environment again (for new graph y) (x will always stay as timestep)
    env = 'Pred-prey miscap0 mixed'
    idx = 2
    ax = None
    ax= plt.subplot(1, 3, idx+1)

    ax.grid()
    # figure = plt.figure()
    # plt.grid()
    method_index = 0

    for (construction, style) in zip(constructions, algorithm_lines):
        x, y, min_length, run_number = read_data_tderror(construction, env, cut_length=501)
        print(env, construction, run_number)

        if run_number == 0:
            continue

        y_mean = smooth_tderror(np.median(y, axis=0))#y_mean = smooth(np.mean(y, axis=0))
        train_scores_mean = y_mean
        data[method_index].append(y_mean[:s_cut])
        method_index += 1

        low = smooth_tderror(np.percentile(y, 25, axis=0))
        high = smooth_tderror(np.percentile(y, 75, axis=0))

        #h = smooth(sem(y) * t.ppf((1 + confidence) / 2, min_length - 1))
        #   h = smooth(sem(data) * t.ppf((1 + confidence) / 2, max_length - 1))
        #bhos = np.linspace(1, min_length, min_length)
        bhos = x[0] / 1000000
        # if log_scale:
        #     train_scores_mean = np.log(train_scores_mean + scale) - np.log(scale)
        #     h = np.log(h + scale) - np.log(scale)
        ax.fill_between(bhos, low,
                            high, alpha=alpha,
                            color=color[construction], linewidth=0)
        width = 4
        ax.plot(bhos, train_scores_mean, color=color[construction], label=construction, linewidth=width, linestyle=style)

    # Others
    ax.tick_params('x', labelsize=font_size)
    ax.tick_params('y', labelsize=font_size)
    ax.set_xlabel('T (mil)', size=font_size)
    ax.set_ylabel('Average TD Error', size=font_size)
    ax.set_title('b) $\mathbf{TD~error}$ on Pred-Prey', size=legend_font_size)
    
    # SMAC environment Specifics
    #if env == 'MMM2':
    #    ax.set_ylim(0.04, 0.26)
    #elif env == '5m_vs_6m':
    #    ax.set_ylim(0.04, 0.72)



    # figure.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=anchor,
    #               prop={'size': legend_font_size}, ncol=min(len(methods), 4), frameon=False)

    figure.tight_layout()
    # plt.show()

    # plt.gca().set_facecolor([248./255, 248./255, 255./255])
    figure.savefig('./meanSmoothing.pdf', bbox_inches='tight', dpi=300)  # , bbox_extra_artists=(lgd,)
    plt.close(figure)