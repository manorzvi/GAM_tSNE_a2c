import numpy as np
import os
import sys
import gc
from pprint import pprint
# sys.path.append(r'C:\Users\Guy\OneDrive - Technion\Project_B\A2C_PHASE_4\baselines')
from a2c_model_inference import a2c_model
import time
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from t_sneV2 import TSNE
import pickle
RS = 20190806

exp_dir       = '/home/manor/anaconda3/envs/openAIenv/baselines/models/a2c2'
obs_dir       = '/home/manor/anaconda3/envs/openAIenv/baselines/models/obs/obs_np'
ckpt_dir      = os.path.join(exp_dir, 'BreakoutNoFrameskip-v4_2e7_a2c_ckpts')
res_dir      = os.path.join(exp_dir, 'res_both_07_dir')
exp_dir       = os.path.abspath(exp_dir)
obs_dir       = os.path.abspath(obs_dir)
ckpt_dir      = os.path.abspath(ckpt_dir)
res_dir       = os.path.abspath(res_dir)
assert os.path.exists(exp_dir)
assert os.path.exists(obs_dir)
assert os.path.exists(ckpt_dir)
assert os.path.exists(res_dir)

mode = 'both'
assert mode in ['reg', 'last', 'prior', 'both']
alpha = 0.7

cwd = os.getcwd()

os.chdir(exp_dir)
print("[t-SNE GEN] - CWD is {}".format(exp_dir))

obs = []
for obs_file in os.listdir(obs_dir):
    if obs_file[-3:]=='npy': # check if it's a file and not a directory
        obs.append(np.load(os.path.join(obs_dir, obs_file)))

print("[t-SNE GEN] - observations fetched:")
print([o.shape for o in obs])

obs = np.concatenate([o for o in obs], axis=0)
print("[t-SNE GEN] - obs shape : {}".format(obs.shape))


num_obs = obs.shape[0]
batch_size = 10000
batches = [obs[i:i + batch_size] for i in range(0, num_obs, batch_size)]

print("[t-SNE GEN] - obs sliced to batches:")
pprint([b.shape for b in batches])

ckpts  = []
tsteps = []
for ckpt_file in os.listdir(ckpt_dir):
    ckpts.append(os.path.join(ckpt_dir, ckpt_file))
    tsteps.append(int(ckpt_file.split('-')[-1]))

ckpts = [x for _,x in sorted(zip(tsteps,ckpts))]

print("[t-SNE GEN] - t-sne mode is {}".format(mode))
if mode in ['prior', 'both']:
    print("[t-SNE GEN] - alpha value: {}".format(alpha))

if mode == 'reg':
    tsne = TSNE(random_state=RS, verbose=2, mode='reg', init='pca')
elif mode == 'last':
    tsne = TSNE(random_state=RS, verbose=2, mode='last', init='pca')
elif mode == 'prior':
    assert alpha is not None
    tsne = TSNE(random_state=RS, verbose=2, mode='prior', alpha=alpha, init='pca')
elif mode == 'both':
    assert alpha is not None
    tsne = TSNE(random_state=RS, verbose=2, mode='both', alpha=alpha, init='pca')

is_pca = True
to_normalize = True
n_components = 50
print("[t-SNE GEN] - pca enabled", end=' ')
if to_normalize:
    print("with normalization")
else:
    print("without normalization")

kl = []
kl_original = []
for ckpt in ckpts:
    print("[t-SNE GEN] - feed-forward model: {}".format(ckpt))
    latent_out_total = []
    value_total      = []
    action_total     = []
    std_total        = []
    neglogp_total    = []
    for b in batches:
        print("       - feed batch size: {} ...".format(b.shape), end=' ')
        model = a2c_model(batch_size=b.shape[0])
        model.load(ckpt)
        start = time.time()
        a, v, neglogp, latent_out, std = model.step(b)
        del model
        action_total.append(a)
        value_total.append(v)
        latent_out_total.append(latent_out)
        std_total.append(std)
        neglogp_total.append(neglogp)
        end = time.time()
        print("done.({})".format(end-start))

    latent_out = np.concatenate([l for l in latent_out_total], axis=0)
    v          = np.concatenate([v for v in value_total], axis=0)
    a          = np.concatenate([a for a in action_total], axis=0)
    std        = np.concatenate([std for std in std_total], axis=0)
    neglogp    = np.concatenate([neglogp for neglogp in neglogp_total], axis=0)
    # latent_out_total_ref = sys.getrefcount(latent_out_total)
    del action_total, value_total, latent_out_total, std_total, neglogp_total
    print("       - total latent activations shape: {}".format(latent_out.shape))
    print("       - total values shape: {}".format(v.shape))
    print("       - total chosen action shape: {}".format(a.shape))
    print("       - total std on actions per obs shape: {}".format(std.shape))
    print("       - total neglogp per obs shape: {}".format(std.shape))
    if is_pca:
        pca = PCA(n_components=n_components)
        if to_normalize:
            latent_out = StandardScaler().fit_transform(latent_out)

        latent_out = pca.fit_transform(latent_out)

    print("       - generate t-SNE for ckpt: {}".format(ckpt))
    embedded = tsne.fit_transform(latent_out)
    kl.append(tsne.kl_divergence_)
    kl_original.append(tsne.kl_divergence_original_)
    os.chdir(res_dir)
    print("       - save latent, value, embedded, std & neflogp... ", end=' ')
    np.savez('{}.npz'.format(os.path.basename(ckpt)), vf=v, lat=latent_out, a=a, kl=kl, embd=embedded, std=std, neglogp=neglogp)
    print("done. delete... ", end=' ')
    os.chdir(exp_dir)
    # latent_out_ref = sys.getrefcount(latent_out)
    del latent_out
    del v
    del a
    del embedded
    del std
    del neglogp
    gc.collect()
    print("done.")

os.chdir(res_dir)
kl          = np.array(kl)
kl_original = np.array(kl_original)
print("       - save kl,kl_original...", end=' ')
np.savez('kl.npz', kl=kl, kl_original=kl_original)
os.chdir(exp_dir)

print("done.")
