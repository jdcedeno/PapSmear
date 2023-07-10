import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.artist as artist
import pickle
import Augmentor
from matplotlib.patches import Rectangle
from skimage.io import imread, imsave
from tkinter.filedialog import askopenfilename


class BrowseImage:
    def __init__(self, fig_frame, toolbar_frame):
        self.img_path = askopenfilename()
        image = imread(self.img_path)

        self.image = np.array(image)

        self.figure = plt.Figure(figsize=(15.8, 7.8))
        self.canvas = FigureCanvasTkAgg(self.figure, fig_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, toolbar_frame)
        self.toolbar.update()

        plt.ion()
        self.img_ax = self.figure.gca()
        if len(np.shape(self.image)) == 3:
            self.img_ax.imshow(self.image)
        else:
            self.img_ax.imshow(self.image, cmap='gray')

        true_list = []
        name = self.img_path
        for count in range(len(name)):
            char = name[count]
            if char == '/':
                true_list.append(count)
        last_dot = true_list[-1]
        name = name[:last_dot]
        print(self.img_path)
        print(name)

        self.rectangles = []
        self.coordinates = []
        self.coords = None
        self.rect = None
        self.key = None
        self.click = False
        self.click_coords_x = None
        self.click_coords_y = None

        self.background = None
        self.pipeline01 = Augmentor.Pipeline(name)
        self.pipeline01.random_distortion(1, 9, 9, 12)

        self.kid = self.figure.canvas.mpl_connect('key_press_event', self.on_key)
        self.krid = self.figure.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self.on_left_click)
        self.cidmotion = self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.crid = self.figure.canvas.mpl_connect('button_release_event', self.on_left_click_release)

    def on_key(self, event):
        if event.inaxes == self.img_ax:
            if event.key == "shift":
                self.key = "shift"
            if event.key == "ctrl+z":
                self.key = "ctrl+z"
                print(self.key)

    def on_key_release(self, event):
        if self.key == "ctrl+z" and len(self.rectangles) >= 1:
            self.rectangles = self.rectangles[:-1]
            self.coordinates = self.coordinates[:-1]
            self.img_ax.clear()
            self.img_ax.imshow(self.image)
            for rect in self.rectangles:
                self.img_ax.add_patch(rect)
                self.img_ax.draw_artist(rect)
            for count in range(len(self.coordinates)):
                center_x = self.coordinates[count]["center_x"]
                center_y = self.coordinates[count]["center_y"]
                self.img_ax.scatter(center_x, center_y, c='r', linewidths=0.1)
            self.canvas.draw()
        self.key = None

    def on_left_click(self, event):
        if event.inaxes == self.img_ax:
            if self.key is "shift":
                self.click = True
                self.click_coords_x = int(event.xdata)
                self.click_coords_y = int(event.ydata)

                self.background = self.canvas.copy_from_bbox(self.img_ax.bbox)

                self.rect = Rectangle((self.click_coords_x, self.click_coords_y), 0, 0, edgecolor='r', fill=None)
                self.rect.set_animated(True)
                self.img_ax.add_patch(self.rect)

    def on_motion(self, event):
        if event.inaxes == self.img_ax:
            if self.key is "shift" and self.click is True:
                x = np.sort([self.click_coords_x, int(event.xdata)])
                y = np.sort([self.click_coords_y, int(event.ydata)])
                width = x[1] - x[0]
                height = y[1] - y[0]

                self.coords = {"x_init": x[0],
                               "y_init": y[0],
                               "x_final": x[1],
                               "y_final": y[1],
                               "center_x": x[0] + int((x[1] - x[0]) / 2),
                               "center_y": y[0] + int((y[1] - y[0]) / 2),
                               "width": width,
                               "height": height}

                self.canvas.restore_region(self.background)

                self.rect.set_x(x[0])
                self.rect.set_y(y[0])
                self.rect.set_width(width)
                self.rect.set_height(height)

                self.img_ax.draw_artist(self.rect)
                self.canvas.blit(self.img_ax.bbox)

    def on_left_click_release(self, event):
        if event.inaxes == self.img_ax:
            if self.click is True:
                self.click = False
                self.rect.set_animated(False)
                self.rectangles.append(self.rect)
                self.coordinates.append(self.coords)
                self.canvas.restore_region(self.background)
                self.img_ax.add_patch(self.rect)
                self.img_ax.draw_artist(self.rect)
                point = self.img_ax.scatter(self.coords["center_x"], self.coords["center_y"], c='r', linewidths=0.1)
                self.img_ax.draw_artist(point)
                self.canvas.blit(self.img_ax.bbox)


class WindowBlueprint(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        hs = self.winfo_screenheight() - 75
        ws = self.winfo_screenwidth() - 10
        self.minsize(width=ws, height=hs)
        self.geometry("{}x{}+0+0".format(ws, hs))

        self.fig_frame = tk.Frame(self, height=720, width=1080)
        self.fig_frame.grid(row=0, column=0)

        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=1, column=0)

        self.toolbar_frame = tk.Frame(self)
        self.toolbar_frame.grid(row=2, column=0)

        self.figure = BrowseImage(self.fig_frame, self.toolbar_frame)
        self.fig_frame.focus_force()

        self.browse_button = tk.Button(self.buttons_frame, text="Browse", command=self.browse, takefocus=False)
        self.browse_button.grid(row=0, column=0)

        self.save_button = tk.Button(self.buttons_frame, text="Save", command=self.save_data, takefocus=False)
        self.save_button.grid(row=0, column=1)

    def browse(self):
        plt.close(self.figure.figure)
        self.figure = BrowseImage(self.fig_frame, self.toolbar_frame)
        self.fig_frame.focus_force()

    def save_data(self):
        true_list = []
        name = self.figure.img_path
        for count in range(len(name)):
            char = name[count]
            if char == '.':
                true_list.append(count)
        last_dot = true_list[-1]
        name = name[:last_dot] + '_box_labels.pickle'
        pickle_out = open(name, "wb")
        pickle.dump(self.figure.coordinates, pickle_out)
        pickle_out.close()
        self.figure.pipeline01.sample(100)


window = WindowBlueprint()
window.mainloop()
pass




























