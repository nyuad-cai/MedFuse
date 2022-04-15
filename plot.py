import matplotlib.pyplot as plt
import torch
import glob
import numpy as np
import re

plt.rcParams.update({'font.size': 30})
# plt.rcParams['figure.figsize'] = [8.0, 3.4]
def plot_types_bar():
    plt.rcParams.update({'font.size': 18})

    keys = ['Acute', 'Mixed', 'Chronic']
    single_aurocs = [0.761, 0.749, 0.717]
    single_aurocs_errors = np.abs(np.array([(0.732, 0.789), (0.724 , 0.773) , (0.693, 0.741)]).T - single_aurocs)
    single_auprcs = [0.432, 0.458, 0.487]
    single_auprcs_errors = np.abs(np.array([(0.386, 0.486) , (0.413,  0.506) , (0.448, 0.530)]).T - single_auprcs)


    multi_aurocs = [0.772, 0.800, 0.745]
    multi_aurocs_errors = np.abs(np.array([(0.744, 0.800) , (0.776 , 0.823) , (0.723, 0.768)]).T - multi_aurocs)
    multi_auprcs = [0.433, 0.565, 0.512]
    multi_auprcs_errors = np.abs(np.array([(0.386, 0.486) , (0.516,  0.614) , (0.473, 0.555)]).T - multi_auprcs)


    barWidth = 0.05
    xs1 = (np.arange(len(single_aurocs)) ) * 0.2
    xs2 = [x + barWidth for x in xs1]

    plt.bar(xs1, single_aurocs, color ='#FF1F5B', width = barWidth,
             label ='Uni-modal')
    plt.bar(xs2, multi_aurocs, color ='#009ADE', width = barWidth,
              label ='Multi-modal')

    plt.errorbar(xs1, single_aurocs, color ='black', yerr = single_aurocs_errors, fmt='o')
    plt.errorbar(xs2, multi_aurocs, color ='black', yerr = multi_aurocs_errors, fmt='o')
    plt.ylabel('AUROC')#, fontweight ='bold', fontsize = 15)
    locs = [(r + barWidth/2) for r in xs1]
    plt.xticks(locs, keys)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylim([0.65, 0.85])  

    plt.legend(loc='upper right')

    plt.savefig(f"plots/bar_aurocs.pdf")
    plt.close()


    plt.bar(xs1, single_auprcs, color ='#FF1F5B', width = barWidth)
    plt.bar(xs2, multi_auprcs, color ='#009ADE', width = barWidth)

    plt.errorbar(xs1, single_auprcs, color ='black', yerr=single_auprcs_errors, fmt='o')
    plt.errorbar(xs2, multi_auprcs, color ='black', yerr=multi_auprcs_errors, fmt='o')

    
    plt.ylabel('AUPRC')#, fontweight ='bold', fontsize = 15)
    locs = [(r + barWidth/2) for r in xs1]
    plt.xticks(locs, keys)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylim([0.35, 0.6])
    
    plt.savefig(f"plots/bar_auprcs.pdf")
    plt.close()

plot_types_bar()


def parse_results(task='pheno', file='results_val.txt'):
    results_auroc = {}
    results_auprc = {}
    paths = glob.glob(f'checkpoints/{task}/ablation2/**/{file}', recursive = True)
    for path in paths:
        file1 = open(path, 'r')
        Lines = file1.readlines()

        bestlines = [line for line in range(len(Lines)) if 'best' in Lines[line] ]

        lines_Selected = Lines[bestlines[-1]]

 
        splited = [(num.split(':')[-1]) for num in lines_Selected.strip().split(' ')]
        splited = [split for split in splited if split !='' and split[0]=='0']
        print(splited)
        # \, splited[5], splited[7])
        auroc = float(splited[1])
        auprc = float(splited[2])

        ratio = float(path.split('/')[-2].split('_')[-1])

        results_auroc[ratio] = auroc
        results_auprc[ratio] = auprc

    keys = results_auprc.keys()
    keys = sorted(list(keys))
    aurocs = [results_auroc[key] for key in keys]
    auprcs = [results_auprc[key] for key in keys]

    return aurocs, auprcs, keys

def plot_aurocs():

    aurocs_mor = [0.868, 0.884, 0.872, 0.873, 0.874, 0.873, 0.876, 0.874, 0.875, 0.872, 0.872]
    aurocs_mor_error = np.abs(np.array([(0.817, 0.901), (0.842, 0.919), (0.833, 0.910), (0.835, 0.906), (0.831, 0.910),  (0.812, 0.911), (0.840, 0.910), (0.840, 0.911), (0.830, 0.911), (0.835, 0.906), (0.840, 0.911) ]).T - aurocs_mor)
    # auprcs_mor_error = [(0.477, 0.714), (0.484, 0.718), (0.468, 0.714), (0.475, 0.703), (0.481, 0.708), (0.458, 0.695), (0.469, 0.701), (0.453, 0.681), (0.452, 0.689), (0.436, 0.669), (0.427, 0.660) ]
    


    # auprcs_pheno = [0.492, 0.491, 0.496, 0.497, 0.495, 0.483, 0.489, 0.476,  0.466, 0.476, 0.466]
    aurocs_pheno = [0.775, 0.773, 0.780, 0.776, 0.776, 0.775, 0.776, 0.771, 0.765, 0.771, 0.765]
    aurocs_pheno_error = np.abs(np.array([(0.736, 0.812),(0.740, 0.810), (0.742, 0.820) , (0.739, 0.816), (0.730, 0.810), (0.736, 0.812), (0.737, 0.812), (0.732, 0.808), (0.726, 0.803), (0.732, 0.808), (0.726, 0.801)]).T - aurocs_pheno)
    
    # plt.rcParams['figure.figsize'] = [8.0, 3.4]
    plt.rcParams.update({'font.size': 12})

    keys = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1.0])*100

    plt.plot(keys, aurocs_mor , label = f"In-hospital mortality", color='#FF1F5B')
    plt.plot(keys, aurocs_pheno , label = f"Phenotyping", color='#009ADE')

    plt.errorbar(keys, aurocs_mor, color ='#FF1F5B', yerr=aurocs_mor_error, fmt='*')
    plt.errorbar(keys, aurocs_pheno, color ='#009ADE', yerr=aurocs_pheno_error, fmt='o')

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylim([0.70, 0.92])
    plt.xticks(keys)
    plt.yticks([0.70, 0.75, 0.80, 0.85, 0.90])
    plt.ylabel('AUROC')
    plt.xlabel('% of uni-modal training samples')

    # plt.legend()
    plt.legend(loc=(.6, .98))
    plt.savefig(f"plots/aurocs.pdf")
    plt.close()
plot_aurocs()

def plot_loss():

    save_dir = 'checkpoints/mmtm_jointonly_a_e2c_on_p/'
    checkpoint = torch.load(f'{save_dir}/last_checkpoint.pth.tar')
    epochs_stats = checkpoint['epochs_stats']

    colors = {
        'loss train joint': '#8a2244', 
        'loss train': '#c687d5', 
        'loss train ehr': '#c687d5',
        'loss train cxr': '#da8c22',  
        'loss val joint': '#80d6f8', 
        'loss val': '#440f06', 
        'loss val ehr' : '#440f06', 
        'loss val cxr': '#000075', 
        'auroc val ehr': '#02a92c', 
        'auroc val': '#02a92c', 
        'auroc val cxr': '#e6194B', 
        'auroc val joint': '#f58231', 
        'auroc val avg': '#ffe119', 
        # '#bfef45'
    }
    keys = [
        'loss train joint', 
    'loss val joint', 
    'auroc val joint'
    ]

    index = 1
    values = ['loss', 'auroc']
    filename = f'{values[index]}.png'
    value = values[index]

    for loss in epochs_stats:
        if loss in keys and value in loss:
            plt.plot(epochs_stats[loss], label = f"{loss.replace('avg', 'late')}", color=colors[loss])
        
    plt.xlabel('epochs')
    plt.title(value)
    plt.legend()
    plt.savefig(f"{save_dir}/{filename}")
    plt.close()
