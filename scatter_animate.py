import matplotlib.pyplot as plt
import matplotlib as m
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

class scatter_animate():
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, embds=None, colors=None, kls=None, frame_annotations=None,
                 filename='Default Scatter Animation', title='Default Title'):

        assert isinstance(embds,np.ndarray)
        assert isinstance(colors, np.ndarray)
        assert isinstance(kls, np.ndarray)

        assert embds.shape[0] == colors.shape[0]
        assert embds.shape[1] == colors.shape[1]
        assert embds.shape[0] == colors.shape[0] == kls.shape[0]


        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        self.embds = embds
        self.kls = kls
        self.numframes = self.embds.shape[0]
        self.colors = colors

        if isinstance(frame_annotations, list):
            self.ann_list = frame_annotations
        else:
            self.ann_list = None

        self.title = title

        self.stream = self.data_stream

        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(fig=self.fig, func=self.update, frames=self.numframes, init_func=self.setup_plot)
        #writer = FFMpegWriter(fps=1, metadata=dict(artist='Me'), bitrate=-1)

        # And save to CWD
        #self.ani.save(filename + '.mp4', writer=writer)
        self.ani.save(filename + '.mp4', fps=1)
        # self.ani.save(filename + '.gif', writer='imagemagick', fps=0.5)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, c = next(self.data_stream(0))

        print("[INFO] - frame {} (setup)".format(0))

        self.scat = self.ax.scatter(x, y, color=c, s=1)

        min_x = x.min()
        min_y = y.min()
        max_x = x.max()
        max_y = y.max()

        self.ax.set_xlim(min_x, max_x)
        self.ax.set_ylim(min_y, max_y)

        self.ax.set_title(self.title)

        if isinstance(self.ann_list, list):
            self.frame_ann = self.ax.annotate("frame: {} ({}/{})".format(self.ann_list[0], 0, self.numframes-1),
                                              xy=(5,5), xycoords='figure points')
        else:
            self.frame_ann = self.ax.annotate("frame: {}/{}".format(0, self.numframes-1),
                                              xy=(5,5), xycoords='figure points')

        self.colorbar = self.fig.colorbar(self.scat)

        return self.scat,

    def data_stream(self,i):
        x = self.embds[i, :, 0]
        y = self.embds[i, :, 1]
        c = self.colors[i, :]
        yield x, y, c

    def update(self,i):
        """Update the scatter plot."""
        x, y, c = next(self.data_stream(i))

        print("[INFO] - frame {} (update)".format(i))

        xy = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
        self.scat.set_offsets(xy) # TODO: check if offsets stands for 'distance from current location' or 'new location'

        min_x = x.min()
        min_y = y.min()
        max_x = x.max()
        max_y = y.max()

        self.ax.set_xlim(min_x, max_x)
        self.ax.set_ylim(min_y, max_y)
        self.scat.set_color(c)
        # Update best value last agent marker

        self.frame_ann.remove()
        if isinstance(self.ann_list,list):
            self.frame_ann = self.ax.annotate("frame: {} ({}/{}) "
                                              "final KL: {:.3f}".format(self.ann_list[i], i, self.numframes-1, self.kls[i]),
                                              xy=(5,5), xycoords='figure points')
        else:
            self.frame_ann = self.ax.annotate("frame: {}/{}".format(i, self.numframes-1),
                                              xy=(5,5), xycoords='figure points')
        return self.scat,