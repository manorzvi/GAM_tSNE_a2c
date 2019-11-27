import os
import pickle
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm


exp_dir  = '/home/manor/anaconda3/envs/openAIenv/baselines/models/BreakoutNoFrameskip-v4_2e7_a2c_exp_2019-06-11_11-00-00'
obs_dir  = '/home/manor/anaconda3/envs/openAIenv/baselines/models/BreakoutNoFrameskip-v4_2e7_a2c_exp_2019-06-11_11-00-00/obs_dir'
ckpt_dir = '/home/manor/anaconda3/envs/openAIenv/baselines/models/BreakoutNoFrameskip-v4_2e7_a2c_exp_2019-06-11_11-00-00/ckpt_dir'

exp_dir = os.path.abspath(exp_dir)
obs_dir = os.path.abspath(obs_dir)
ckpt_dir = os.path.abspath(ckpt_dir)

assert os.path.exists(exp_dir)
assert os.path.exists(obs_dir)
assert os.path.exists(ckpt_dir)

os.chdir(exp_dir)

latents      = []
agent_idx   = []
agent_names = []
for data_file in os.listdir(exp_dir):
    if not data_file.endswith('.data'):
        continue
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    latents.append(data['lat'])
    agent_names.append(data_file[:-5])
    agent_idx.append(int(data_file[:-5].split('-')[-1]))
    print("[ESTIMATE] - fetched: {}".format(data_file))

print("\n[ESTIMATE] - re-order data according to agents indices... ",end='')
latents    = [x for _, x in sorted(zip(agent_idx, latents))]
agent_names     = [x for _, x in sorted(zip(agent_idx, agent_names))]
print("done.")


print("\n[ESTIMATE] - stack data... ",end='')
latents = np.stack(latents, axis=0)
print("done.")

fig2,ax = plt.subplots()

for i in range(latents.shape[0]):

    sample = latents[i,:,:]
    mean = np.mean(sample,axis=0)
    sample = sample.T
    s = np.linalg.svd(sample, full_matrices=True, compute_uv=0)
    s_square = np.power(s,2)
    s_square_cumsum = np.cumsum(s_square)
    rk = s_square_cumsum/s_square_cumsum[-1]

    dims = list(range(1, rk.shape[0]+1))

    ax.plot(dims, rk, label=agent_names[i])
    #ax.annotate("({},{:.3f})".format(dims[1],rk[1]), xy=(dims[1],rk[1]))
    #ax.annotate("({},{:.3f})".format(dims[2],rk[2]), xy=(dims[2],rk[2]))
    print("[ESTIMATE] - plot: {}".format(agent_names[i]))

ax.legend()
plt.show()








