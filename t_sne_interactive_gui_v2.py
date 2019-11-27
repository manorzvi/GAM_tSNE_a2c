import os
import re
import matplotlib.pyplot  as plt
import numpy              as np
from pprint               import pprint
from matplotlib.widgets   import Slider, Button, RadioButtons
from matplotlib.widgets   import LassoSelector
from matplotlib.path      import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.
    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.
    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to interact with.
    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.5):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other
        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_edgecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        lineprops = {'color': 'red', 'linewidth': 2, 'alpha': 0.8}
        self.lasso = LassoSelector(ax=ax, onselect=self.onselect, lineprops=lineprops)
        self.ind = np.array([])

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc = self.collection.get_edgecolor()
        if np.any(self.ind):
            self.fc[:, -1] = self.alpha_other
            self.fc[self.ind, -1] = 1
            self.collection.set_facecolors(self.fc)
            self.canvas.draw_idle()
        else:
            self.fc[:, -1] = 1
            self.collection.set_facecolors(self.fc)
            self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

def custom_colormap(array, repeats=None):
    print(array.shape)

    if len(array.shape) == 1: # Hand crafted features (exist per observations)
        assert repeats is not None
        x = array.shape[0]
        colormap_t = np.zeros((x, 4))
        colormap_t[:,0] = array
        colormap_t[:,2] = array
        colormap_t[:,3] = 1
        colormap =  np.zeros((repeats, x, 4))
        for i in range(repeats):
            colormap[i,:,:] = colormap_t
    elif len(array.shape) == 2: # Model features (exists per checkpoint of training)
        y, x = array.shape
        colormap = np.zeros((y, x, 4))
        for i in range(y):
            colormap[i,:,0] = array[i,:]
            colormap[i,:,2] = array[i, :]
            colormap[i,:,3] = 1

    return colormap

def get_tsne_colormap(agent_fdir, obs_fdir, experiment):
    vfs     = []
    std     = []
    actions = []
    for ckpt_data in agent_fdir[experiment]:
        ckpt_data = os.path.join(agent_fdir['fullpath'], experiment, ckpt_data)
        data = np.load(ckpt_data)
        vfs.append(data['vf'])
        std.append(data['std'])
        actions.append(data['a'])

    # value function normalization:
    vfs = [(x - np.amin(x)) / (np.amax(x) - np.amin(x)) for x in vfs]
    vfs = np.stack(vfs,axis=0)

    # outputs std normalizations:
    std = [(x - np.amin(x)) / (np.amax(x) - np.amin(x)) for x in std]
    std = np.stack(std, axis=0)

    # actions normalizations:
    actions = [x/3 for x in actions]
    actions = np.stack(actions, axis=0)

    colormap = plt.get_cmap()

    vfs_colormap = colormap(vfs)
    # vfs_colormap = custom_colormap(vfs)
    std_colormap = colormap(std)
    # std_colormap = custom_colormap(std)
    actions_colormap = colormap(actions)
    # actions_colormap = custom_colormap(actions)

    padel_positions_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'padel_positions.npy')
    print('\t\tLoad Padel Positions: {}...'.format(padel_positions_f), end=' ')
    try:
        padel = np.load(padel_positions_f)
        padel_n = (padel - padel.min()) / (padel.max() - padel.min())
        padel_n = np.repeat(np.expand_dims(padel_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("\t\tNo such file or directory: {} - set same as vfs.".format(padel_positions_f))
        padel_n = vfs

    padel_colormap = colormap(padel_n)
    # padel_colormap = custom_colormap(padel_n, repeats=vfs.shape[0])
    print()

    ball_positions_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'ball_positions.npz')
    print('\t\tLoad Ball Positions: {}...'.format(ball_positions_f), end=' ')
    try:
        ball = np.load(ball_positions_f)
        ball_areas = ball['areas'].astype(np.int32)
        ball_areas_n = (ball_areas - ball_areas.min()) / (ball_areas.max() - ball_areas.min())
        ball_areas_n = np.repeat(np.expand_dims(ball_areas_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("\t\tNo such file or directory: {} - set same as vfs.".format(ball_positions_f))
        ball_areas_n = vfs

    ball_colormap = colormap(ball_areas_n)
    # ball_colormap = custom_colormap(ball_areas_n, repeats=vfs.shape[0])
    print()

    bricks_count_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'bricks_count.npy')
    print('\t\tLoad Bricks Count: {}...'.format(bricks_count_f), end=' ')
    try:
        bricks = np.load(bricks_count_f)
        bricks_n = (bricks - bricks.min()) / (bricks.max() - bricks.min())
        bricks_n = np.repeat(np.expand_dims(bricks_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("\t\tNo such file or directory: {} - set same as vfs.".format(bricks_count_f))
        bricks_n = vfs

    bricks_colormap = colormap(bricks_n)
    # bricks_colormap = custom_colormap(bricks_n, repeats=vfs.shape[0])
    print()

    tunnel_depth_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'tunnel_depth.npy')
    print('\t\tLoad Tunnel Depth: {}...'.format(tunnel_depth_f), end=' ')
    try:
        tunnel_depth = np.load(tunnel_depth_f)
        tunnel_depth_n = (tunnel_depth - tunnel_depth.min()) / (tunnel_depth.max() - tunnel_depth.min())
        tunnel_depth_n = np.repeat(np.expand_dims(tunnel_depth_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("\t\tNo such file or directory: {} - set same as vfs.".format(tunnel_depth_f))
        tunnel_depth_n = vfs

    tunnel_depth_colormap = colormap(tunnel_depth_n)
    # tunnel_depth_colormap = custom_colormap(tunnel_depth_n, repeats=vfs.shape[0])
    print()

    tunnel_isopen_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'tunnel_is_open.npy')
    print('\t\tLoad Tunnel Is Open: {}...'.format(tunnel_isopen_f), end=' ')
    try:
        tunnel_is_open = np.load(tunnel_isopen_f)
        tunnel_is_open = tunnel_is_open.astype(int)
        # tunnel_is_open = np.repeat(np.expand_dims(tunnel_is_open, axis=0), vfs.shape[0], axis=0)
    except:
        print("\t\tNo such file or directory: {} - set same as vfs.".format(tunnel_isopen_f))
        tunnel_is_open = vfs

    # tunnel_is_open_colormap = colormap(tunnel_is_open)
    tunnel_is_open_colormap = custom_colormap(tunnel_is_open, repeats=vfs.shape[0])
    print()

    score_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'score.npy')
    print('\t\tLoad Score: {}...'.format(score_f), end=' ')
    try:
        score = np.load(score_f)
        score_n = (score - score.min()) / (score.max() - score.min())
        score_n = np.repeat(np.expand_dims(score_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("\t\tNo such file or directory: {} - set same as vfs.".format(score_f))
        score_n = vfs

    score_colormap = colormap(score_n)
    # score_colormap = custom_colormap(score_n, repeats=vfs.shape[0])
    print()

    lives_f = os.path.join(obs_fdir['fullpath'], 'features_hc', 'lives.npy')
    print('\t\tLoad Lives: {}...'.format(lives_f), end=' ')
    try:
        lives = np.load(lives_f)
        lives_n = (lives - lives.min()) / (lives.max() - lives.min())
        lives_n = np.repeat(np.expand_dims(lives_n, axis=0), vfs.shape[0], axis=0)
    except:
        print("\t\tNo such file or directory: {} - set same as vfs.".format(lives_f))
        lives_n = vfs

    lives_colormap = colormap(lives_n)
    # lives_colormap = custom_colormap(lives_n, repeats=vfs.shape[0])
    print()

    colormaps = {'vfs'          : vfs_colormap,
                 'std'          : std_colormap,
                 'actions'      : actions_colormap,
                 'padel'        : padel_colormap,
                 'ball'         : ball_colormap,
                 'bricks'       : bricks_colormap,
                 'tunnel_open'  : tunnel_is_open_colormap,
                 'tunnel_depth' : tunnel_depth_colormap,
                 'score'        : score_colormap,
                 'lives'        : lives_colormap}

    return colormaps

def get_tsne_embd(agent_fdir, experiment):
    embd_2d = []
    for ckpt_data in agent_fdir[experiment]:
        ckpt_data = os.path.join(agent_fdir['fullpath'], experiment, ckpt_data)
        data = np.load(ckpt_data)
        embd_2d.append(data['embd'])

    embd_2d = np.stack(embd_2d, axis=0)

    return embd_2d

def get_agnts_experiment_min_max(embd):
    max_x, max_y = np.max(embd, axis=0)
    min_x, min_y = np.min(embd, axis=0)

    return min_x, min_y, max_x, max_y

def get_obs_and_lengths(obs_fdir):
    obs = []
    lengths = []

    for obs_file in obs_fdir['obs_np']:
        obs_file = os.path.join(obs_fdir['fullpath'], 'obs_np', obs_file)
        obs.append(np.load(obs_file))

    obs = np.concatenate([o for o in obs], axis=0)

    for length in obs_fdir['obs_lengths']:
        lengths.append(length)

    return obs, lengths

def get_tsne_titles_and_final_kl(agent_fdir, experiment):

    kl = []
    intervals = []

    kl = np.load(os.path.join(agent_fdir['fullpath'], experiment, 'kl.npz'))['kl_original']

    for ckpt_data in agent_fdir[experiment]:
        if ckpt_data.startswith('Breakout'):
            ckpt_re = re.match('^.*ckpts-(\d+).npz$', ckpt_data)
            intervals.append(int(ckpt_re.group(1)))

    return kl, intervals

class t_sne_interactive_gui:

    def __init__(self, models_database, init_agent='a2c1', init_experiment='res_reg_dir', point_size=3):
        assert isinstance(models_database, dict)

        self.model_database = models_database

        print('Load colormaps of Agent: {}, Experiment: {}...'.format(init_agent, init_experiment))
        self.colors = get_tsne_colormap(agent_fdir=models_database[init_agent],
                                        obs_fdir=models_database['obs'],
                                        experiment=init_experiment)
        print('Done.')

        print('Load embedded data of Agent: {}, Experiment: {}...'.format(init_agent, init_experiment))
        self.embd  = get_tsne_embd(agent_fdir=models_database[init_agent], experiment=init_experiment)
        print('Done. (self.embd.shape={})'.format(self.embd.shape))

        print('Load Observations: {}...'.format(models_database['obs']['fullpath']))
        self.obs, self.lengths = get_obs_and_lengths(models_database['obs'])
        print('Done.')
        # TODO: combine with get_tsne_colormap later
        self.colors['timesteps'] = np.zeros(self.colors['vfs'].shape)
        j = 0
        for length in self.lengths:
            for i in range(length):
                self.colors['timesteps'][:,j+i,1] = i
                self.colors['timesteps'][:,j+i,3] = 1
            j += i+1

        for i in range(self.colors['timesteps'].shape[1]):
            self.colors['timesteps'][:,i,1] = (self.colors['timesteps'][0,i,1] - self.colors['timesteps'][0,:,1].min()) / \
                                              (self.colors['timesteps'][0,:,1].max() - self.colors['timesteps'][0,:,1].min())

        print('Load KL Loss Values & Titles of Agent: {}, Experiment: {}...'.format(init_agent, init_experiment))
        self.agent_experiment_kl, self.agent_experiment_intervals = get_tsne_titles_and_final_kl(agent_fdir=models_database[init_agent], experiment=init_experiment)
        self.current_kl = self.agent_experiment_kl[0]
        self.current_interval = self.agent_experiment_intervals[0]
        print('Done.')

        self.min_x, self.min_y, self.max_x, self.max_y = get_agnts_experiment_min_max(embd=self.embd[0,:,:])

        self.fig, self.ax = plt.subplots()

        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.8, top=0.95)
        self.scat = plt.scatter(x=self.embd[0,:,0],
                                y=self.embd[0,:,1],
                                color=self.colors['vfs'][0,:], # By default data is value-function labeled
                                s=point_size)
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)

        self.fig.suptitle('Latent Space tSNE Projection of {}/{} Training Interval {} (KL={:.4f})'.format(init_agent,
                                                                                                 init_experiment,
                                                                                                 self.current_interval,
                                                                                                 self.current_kl))

        self.point_ind_text = plt.text(self.ax.get_xlim()[0], self.ax.get_ylim()[0], "", fontsize=8)

        # self.colorbar = plt.colorbar(self.scat)

        # Slider
        self.slider_color = 'lightgoldenrodyellow'
        self.current_slider_location = 0
        self.slider_ax    = plt.axes([0.075, 0.05, 0.85, 0.025], facecolor=self.slider_color)
        self.slider       = Slider(self.slider_ax, 'Frame#', 0, self.embd.shape[0] - 1,
                                   valinit=0, valfmt='%0.0f')
        self.slider.on_changed(self.update)

        # Lasso
        self.selector = SelectFromCollection(self.ax, self.scat, alpha_other=0.01)

        self.legend_color = 'lightgoldenrodyellow'
        self.rax = plt.axes([0, 0.15, 0.10, 0.8], facecolor=self.legend_color)
        self.radio = RadioButtons(self.rax, ('value', 'std', 'actions', 'padel', 'ball', 'bricks',
                                             'tunnel_open', 'tunnel_depth', 'score', 'lives', 'timesteps'), active=0) # the index of value button is 0
        self.current_label = 'value'
        self.radio.on_clicked(self.colorfunc)

        self.ra_agents = plt.axes([0.85, 0.15, 0.15, 0.8], facecolor=self.legend_color)
        self.agents, active = self.get_agents(init_agent, init_experiment)
        self.current_agent = init_agent
        self.current_experiment = init_experiment
        print(self.agents, active)
        self.agents_buttons=RadioButtons(self.ra_agents, self.agents, active=active)
        self.agents_buttons.on_clicked(self.agentfunc)

        # create frame navigation buttons
        self.prev_next_buttons_color = 'lightgoldenrodyellow'
        self.nextFrm_ax = plt.axes([0.15, 0.025, 0.15, 0.025])
        self.buttonNext = Button(self.nextFrm_ax, 'Next Frame', color=self.prev_next_buttons_color, hovercolor='0.975')
        self.buttonNext.on_clicked(self.NextFrame)

        self.prevFrm_ax = plt.axes([0.7, 0.025, 0.15, 0.025])
        self.buttonPrev = Button(self.prevFrm_ax, 'Prev Frame', color=self.prev_next_buttons_color, hovercolor='0.975')
        self.buttonPrev.on_clicked(self.PrevFrame)

        # set image visualization, connect image to coordinate
        # create the annotations box
        self.im = OffsetImage(self.obs[0, :, :, 0], zoom=2)
        self.xybox = (50., 50.)
        self.ab = AnnotationBbox(self.im, (0, 0), xybox=self.xybox, xycoords='data',
                                 boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))
        # add it to the axes and make it invisible
        self.ax.add_artist(self.ab)
        self.ab.set_visible(False)

        # add callback for mouse moves
        self.ann = None
        self.fig.canvas.mpl_connect('button_press_event', self.click_func)

        # self.fig.canvas.mpl_connect('motion_notify_event', self.hover)

    def get_agents(self, init_agent, init_experiment):
        agents = list()
        for agent, dict in self.model_database.items():
            if not agent.startswith('a2c'): continue
            for res in dict.keys():
                if res.startswith('res_'):
                    agents.append(agent + '_' + res)
        init_agent_experiment = init_agent + '_' + init_experiment
        for i, agent in enumerate(agents):
            if agent == init_agent_experiment:
                active = i
        return agents, active

    def update(self, val):
        print('update: ' + str(int(np.round(self.slider.val))))

        self.scat.set_offsets(self.embd[int(np.round(self.slider.val)), :, :].tolist())

        self.min_x, self.min_y, self.max_x, self.max_y = \
            get_agnts_experiment_min_max(embd=self.embd[int(np.round(self.slider.val)), :, :])
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)

        self.selector.xys = self.scat.get_offsets()

        self.current_interval = self.agent_experiment_intervals[int(np.round(self.slider.val))]
        self.current_kl       = self.agent_experiment_kl[int(np.round(self.slider.val))]
        self.current_slider_location = int(np.round(self.slider.val))
        self.fig.suptitle('Latent Space tSNE Projection of {}/{}'
                          'Training Interval {} (KL={:.4f})'.format(self.current_agent,self.current_experiment,
                                                                    self.current_interval,
                                                                    self.current_kl))
        self.colorfunc(label=self.current_label)

        self.fig.canvas.draw_idle()

    def agentfunc(self, val):
        print('enter agentfunc')
        assert val in self.agents
        experiment = '_'.join(val.split('_')[1:])
        agent = val.split('_')[0]

        print('Load colormaps of Agent: {}, Experiment: {}...'.format(agent, experiment))
        self.colors = get_tsne_colormap(agent_fdir=self.model_database[agent],
                                        obs_fdir=self.model_database['obs'],
                                        experiment=experiment)
        print('Done.')
        # TODO: combine with get_tsne_colormap later
        self.colors['timesteps'] = np.zeros(self.colors['vfs'].shape)
        j = 0
        for length in self.lengths:
            for i in range(length):
                self.colors['timesteps'][:, j + i, 1] = i
                self.colors['timesteps'][:, j + i, 3] = 1
            j += i + 1

        for i in range(self.colors['timesteps'].shape[1]):
            self.colors['timesteps'][:, i, 1] = (self.colors['timesteps'][0, i, 1] - self.colors['timesteps'][0, :,1].min()) / \
                                                (self.colors['timesteps'][0, :, 1].max() - self.colors['timesteps'][0,:, 1].min())

        print('Load embedded data of Agent: {}, Experiment: {}...'.format(agent, experiment))
        self.embd = get_tsne_embd(agent_fdir=self.model_database[agent], experiment=experiment)
        print('Done. (self.embd.shape={})'.format(self.embd.shape))

        self.scat.set_offsets(self.embd[int(np.round(self.slider.val)), :, :].tolist())
        self.selector.xys = self.scat.get_offsets()

        self.min_x, self.min_y, self.max_x, self.max_y = \
            get_agnts_experiment_min_max(embd=self.embd[int(np.round(self.slider.val)), :, :])
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)

        print('Load KL Loss Values & Titles of Agent: {}, Experiment: {}...'.format(agent, experiment))
        self.agent_experiment_kl, self.agent_experiment_intervals = get_tsne_titles_and_final_kl(
            agent_fdir=self.model_database[agent], experiment=experiment)
        self.current_kl = self.agent_experiment_kl[self.current_slider_location]
        self.current_interval = self.agent_experiment_intervals[self.current_slider_location]
        self.current_agent = agent
        self.current_experiment = experiment
        print('Done.')

        self.fig.suptitle('Latent Space tSNE Projection of {}/{}'
                          'Training Interval {} (KL={:.4f})'.format(self.current_agent, self.current_experiment,
                                                                    self.current_interval,
                                                                    self.current_kl))
        self.fig.canvas.draw_idle()

        self.colorfunc(label=self.current_label)

    def colorfunc(self, label):
        switcher = {'value':          self.colors['vfs'],
                    'std' :           self.colors['std'],
                    'actions' :       self.colors['actions'],
                    'padel':          self.colors['padel'],
                    'ball' :          self.colors['ball'],
                    'bricks':         self.colors['bricks'],
                    'tunnel_open' :   self.colors['tunnel_open'],
                    'tunnel_depth' :  self.colors['tunnel_depth'],
                    'score'        :  self.colors['score'],
                    'lives'        :  self.colors['lives'],
                    'timesteps'    :  self.colors['timesteps']
                    }
        prev_colors = self.scat.get_edgecolor()
        print('colorfunc: ' + str(int(np.round(self.slider.val))))
        print('colorfunc label: ' + label)

        color_array = switcher.get(label, "Invalid Color")[int(np.round(self.slider.val)), :, :]
        color_array[:, -1] = prev_colors[:, -1]  # preserve selection
        self.scat.set_color(color_array)
        self.current_label = label
        self.fig.canvas.draw_idle()

    def NextFrame(self, event):
        currentFrame = int(np.round(self.slider.val))
        if currentFrame < self.slider.valmax:
            self.slider.val = currentFrame + 1
            self.update(currentFrame + 1)
            self.slider.set_val(currentFrame + 1)

    def PrevFrame(self, event):
        currentFrame = int(np.round(self.slider.val))
        if currentFrame > 0:
            self.slider.val = currentFrame - 1
            self.update(currentFrame - 1)
            self.slider.set_val(currentFrame - 1)

    def click_func(self, event):
        if event.button == 3:
            # if the mouse is over the scatter points
            if self.scat.contains(event)[0]:
                # find out the index within the array from the event
                ind = self.scat.contains(event)[1]["ind"][0]
                print(ind)
                # get the figure size
                w, h = self.fig.get_size_inches() * self.fig.dpi
                ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
                hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
                # if event occurs in the top or right quadrant of the figure,
                # change the annotation box position relative to mouse.
                self.ab.xybox = (self.xybox[0] * ws, self.xybox[1] * hs)
                # make annotation box visible
                self.ab.set_visible(True)
                # place it at the position of the hovered scatter point
                self.ab.xy = (self.scat.get_offsets()[ind, 0], self.scat.get_offsets()[ind, 1])

                if (self.selector.ind.size != 0) and (ind in self.selector.ind):  # there are marked points, so show avg. image
                    avg_image = np.mean(self.obs[self.selector.ind, :, :, -1], axis=(0))
                    self.im.set_data(avg_image)
                else:
                    # set the image corresponding to that point
                    self.im.set_data(self.obs[ind, :, :, -1])

            else:
                # if the mouse is not over a scatter point
                self.ab.set_visible(False)

            self.fig.canvas.draw_idle()

    def hover(self, event):
        self.fig.canvas.draw_idle()
        # if the mouse is over the scatter points
        if self.scat.contains(event)[0]:
            # find out the index within the array from the event
            ind = self.scat.contains(event)[1]["ind"][0]
            self.point_ind_text.set_text(ind)
            self.fig.canvas.draw_idle()
        else:
            self.point_ind_text.set_text("")
            self.fig.canvas.draw_idle()

def walk_through_obs(obs_dir_dict):
    assert isinstance(obs_dir_dict, dict)

    obs_dir_dict['obs_np'] = [f for f in os.listdir(os.path.join(obs_dir_dict['fullpath'], 'obs_np'))]
    obs_dir_dict['obs_lengths'] = [len(np.load(os.path.join(obs_dir_dict['fullpath'], 'obs_np', obs))) for
                                   obs in obs_dir_dict['obs_np']]

    obs_dir_dict['obs_lengths'] = [x for _, x in sorted(zip(obs_dir_dict['obs_np'],
                                                            obs_dir_dict['obs_lengths']))]
    obs_dir_dict['obs_np'].sort()

    obs_dir_dict['features_hc'] = [f for f in os.listdir(os.path.join(obs_dir_dict['fullpath'], 'features_hc'))]

def walk_through_agent(agent_dir_dict):
    assert isinstance(agent_dir_dict, dict)

    for d in os.listdir(agent_dir_dict['fullpath']):

        if d.startswith('BreakoutNoFrameskip'):
            agent_dir_dict['checkpoints_path'] = d
            agent_ckpts = [f for f in os.listdir(os.path.join(agent_dir_dict['fullpath'], d))]
            assert len(agent_ckpts) > 0
            agent_indices = []
            for ckpt in agent_ckpts:
                ckpt_re = re.match('^.*ckpts-(\d+)$', ckpt)
                agent_indices.append(int(ckpt_re.group(1)))
            agent_ckpts = [x for _, x in sorted(zip(agent_indices, agent_ckpts))]
            agent_dir_dict['checkpoints_fnames'] = agent_ckpts

        if d.startswith('res_'):
            agent_ckpts = [f for f in os.listdir(os.path.join(agent_dir_dict['fullpath'], d)) if not f.startswith('kl')]
            assert len(agent_ckpts) > 0
            agent_indices = []
            for ckpt in agent_ckpts:
                ckpt_re = re.match('^.*ckpts-(\d+).npz$', ckpt)
                agent_indices.append(int(ckpt_re.group(1)))
            agent_ckpts = [x for _, x in sorted(zip(agent_indices, agent_ckpts))]
            agent_dir_dict[d] = agent_ckpts

def walk_through_models(models_dir):
    assert os.path.exists(models_dir)

    agent_dirs = {}
    for agent_dir in os.listdir(models_dir):

        if agent_dir == 'obs':
            agent_dirs['obs'] = {}
            agent_dirs['obs']['fullpath'] = os.path.join(models_dir, 'obs')
            walk_through_obs(agent_dirs['obs'])

            assert len(agent_dirs['obs']['obs_np']) != 0
            assert len(agent_dirs['obs']['features_hc']) != 0

        if agent_dir.startswith('a2c'):
            agent_dirs[agent_dir] = {}
            agent_dirs[agent_dir]['fullpath'] = os.path.join(models_dir, agent_dir)
            walk_through_agent(agent_dirs[agent_dir])

    return agent_dirs

def main():

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

    gui = t_sne_interactive_gui(models_database=models_database, init_agent='a2c1',
                                init_experiment='res_reg_dir')
    plt.show()

if __name__ == '__main__':
    main()
