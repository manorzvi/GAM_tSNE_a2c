import os
import pickle
import numpy as np
from scatter_animate import *
from pprint import pprint
from t_sne_interactive_gui_v2 import walk_through_models

def get_colormaps(agent_fdir, obs_fdir, experiment):

    colormap = plt.get_cmap()

    vfs = []
    for ckpt_file in models_database[agent][experiment]:
        ckpt_file = os.path.join(models_database[agent]['fullpath'], experiment, ckpt_file)
        ckpt_data = np.load(ckpt_file)
        vf = ckpt_data['vf']
        vfs.append(vf)

    # value function normalization:
    vfs = [(x - np.amin(x)) / (np.amax(x) - np.amin(x)) for x in vfs]
    vfs = np.stack(vfs, axis=0)
    vfs_colormap = colormap(vfs)

    padel_positions_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'padel_positions.npy')
    print('Load Padel Positions: {}...'.format(padel_positions_f))
    try:
        padel = np.load(padel_positions_f)
        padel_n = (padel - padel.min()) / (padel.max() - padel.min())
        padel_n = np.repeat(np.expand_dims(padel_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("No such file or directory: {} - set same as vfs.".format(padel_positions_f))
        padel_n = vfs
    padel_colormap = colormap(padel_n)

    ball_positions_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'ball_positions.npz')
    print('Load Ball Positions: {}...'.format(ball_positions_f))
    try:
        ball = np.load(ball_positions_f)
        ball_areas = ball['areas'].astype(np.int32)
        ball_areas_n = (ball_areas - ball_areas.min()) / (ball_areas.max() - ball_areas.min())
        ball_areas_n = np.repeat(np.expand_dims(ball_areas_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("No such file or directory: {} - set same as vfs.".format(ball_positions_f))
        ball_areas_n = vfs
    ball_colormap = colormap(ball_areas_n)

    bricks_count_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'bricks_count.npy')
    print('Load Bricks Count: {}...'.format(bricks_count_f))
    try:
        bricks = np.load(bricks_count_f)
        bricks_n = (bricks - bricks.min()) / (bricks.max() - bricks.min())
        bricks_n = np.repeat(np.expand_dims(bricks_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("No such file or directory: {} - set same as vfs.".format(bricks_count_f))
        bricks_n = vfs
    bricks_colormap = colormap(bricks_n)

    tunnel_depth_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'tunnel_depth.npy')
    print('Load Tunnel Depth: {}...'.format(tunnel_depth_f))
    try:
        tunnel_depth = np.load(tunnel_depth_f)
        tunnel_depth_n = (tunnel_depth - tunnel_depth.min()) / (tunnel_depth.max() - tunnel_depth.min())
        tunnel_depth_n = np.repeat(np.expand_dims(tunnel_depth_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("No such file or directory: {} - set same as vfs.".format(tunnel_depth_f))
        tunnel_depth_n = vfs
    tunnel_depth_colormap = colormap(tunnel_depth_n)

    tunnel_isopen_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'tunnel_is_open.npy')
    print('Load Tunnel Is Open: {}...'.format(tunnel_isopen_f))
    try:
        tunnel_is_open = np.load(tunnel_isopen_f)
        tunnel_is_open = tunnel_is_open.astype(int) *0.99
        tunnel_is_open = np.repeat(np.expand_dims(tunnel_is_open, axis=0), vfs.shape[0], axis=0)
    except:
        print("No such file or directory: {} - set same as vfs.".format(tunnel_isopen_f))
        tunnel_is_open = vfs
    tunnel_is_open_colormap = colormap(tunnel_is_open)

    score_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'score.npy')
    print('Load Score: {}...'.format(score_f))
    try:
        score = np.load(score_f)
        score_n = (score - score.min()) / (score.max() - score.min())
        score_n = np.repeat(np.expand_dims(score_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("No such file or directory: {} - set same as vfs.".format(score_f))
        score_n = vfs
    score_colormap = colormap(score_n)

    lives_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'lives.npy')
    print('Load Lives: {}...'.format(lives_f))
    try:
        lives = np.load(lives_f)
        lives_n = (lives - lives.min()) / (lives.max() - lives.min())
        lives_n = np.repeat(np.expand_dims(lives_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("No such file or directory: {} - set same as vfs.".format(lives_f))
        lives_n = vfs
    lives_colormap = colormap(lives_n)

    colormaps = {
        'vfs':             vfs_colormap,
        'padel':           padel_colormap,
        'ball':            ball_colormap,
        'bricks':          bricks_colormap,
        'tunnel_open':     tunnel_is_open_colormap,
        'tunnel_depth':    tunnel_depth_colormap,
        'score':           score_colormap,
        'lives':           lives_colormap
    }

    return colormaps

models_dir = 'baselines/models'
models_dir = os.path.abspath(models_dir)
assert os.path.exists(models_dir)

obs_dir = os.path.join(models_dir, 'obs')
obs_dir = os.path.abspath(obs_dir)
assert os.path.exists(obs_dir)

handcrafted_features_dir = os.path.join(obs_dir, 'features_hc')
handcrafted_features_dir = os.path.abspath(handcrafted_features_dir)
assert os.path.exists(handcrafted_features_dir)

models_database = walk_through_models(models_dir)
pprint(models_database)
print('|' + '=' * 80 + '|')

agent = 'a2c1'
experiment = 'res_last_dir'
color = 'tunnel_open'

colormaps = get_colormaps(models_database[agent], models_database['obs'], experiment)

embds   = []
names   = []
indices = []

for ckpt_file in models_database[agent][experiment]:
    ckpt_file = os.path.join(models_database[agent]['fullpath'], experiment, ckpt_file)
    ckpt_data = np.load(ckpt_file)

    embd = ckpt_data['embd']
    name = os.path.basename(ckpt_file)[:-4]
    indice = int(name.split('-')[-1])

    embds.append(embd)
    names.append(name)
    indices.append(indice)

embds = np.stack(embds, axis=0)


kls_file = os.path.join(models_database[agent]['fullpath'], experiment, 'kl.npz')
assert os.path.exists(kls_file)
kls = np.load(kls_file)
kls = kls['kl_original']

kls = np.stack(kls, axis=0)

print(kls.shape)
print(embds.shape)
print(colormaps[color].shape)

gif_name = agent+'_'+experiment+'_'+color

anim = scatter_animate(embds=embds, colors=colormaps[color], kls=kls,
                       filename=gif_name, frame_annotations=names, title=gif_name)
