import tkinter as tk
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
from skimage.io import imread
from PapSmear import ContrastEnhancement as ce
from skimage.color import rgb2gray


class Previewer(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)
        hs = self.winfo_screenheight() - 75
        ws = self.winfo_screenwidth() - 10
        self.minsize(width=ws, height=hs)
        self.geometry("{}x{}+0+0".format(ws, hs))

        self.figure_frame = tk.Frame(self)
        self.figure_frame.pack(fill="x", expand=True)

        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack(fill="x")

        self.toolbar_frame = tk.Frame(self)
        self.toolbar_frame.pack(fill="x")

        self.fig = plt.figure(figsize=(15.9, 6))
        plt.ion()
        self.o_im_axes = self.fig.add_axes([0.02, 0.08, 0.47, 0.775])
        self.o_im_axes.set_title("Original Image")
        self.n_im_axes = self.fig.add_axes([0.518, 0.08, 0.47, 0.775])
        self.n_im_axes.set_title("Preview")
        self.canvas = FigureCanvasTkAgg(self.fig, self.figure_frame)
        self.canvas.get_tk_widget().pack()
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.browse_button = tk.Button(self.buttons_frame, text="Browse", command=self.browse_image)
        self.browse_button.grid(row=0, column=0, padx=3, pady=3)
        self.original_image = None

        self.nint_button = tk.Button(self.buttons_frame, text="Nint", command=self.nint_enhancement, state="disabled")
        self.nint_button.grid(row=0, column=1, padx=3, pady=3)

        self.gauss_button = tk.Button(self.buttons_frame, text="Gauss", command=self.gauss_enhancement, state="disabled")
        self.gauss_button.grid(row=0, column=2, padx=3, pady=3)

        self.root_button = tk.Button(self.buttons_frame, text="Root", command=self.root_enhancement, state="disabled")
        self.root_button.grid(row=0, column=4, padx=3, pady=3)

        self.close_button = tk.Button(self.buttons_frame, text="Close", command=self.close_window)
        self.close_button.grid(row=0, column=5, padx=3, pady=3)

        self.preview_image = None

    def browse_image(self):
        image_path = askopenfilename()
        self.original_image = imread(image_path)
        self.o_im_axes.imshow(self.original_image)

        self.nint_button.configure(state="active")
        self.gauss_button.configure(state="active")
        self.root_button.configure(state="active")

    def close_window(self):
        plt.close('all')
        self.destroy()

    @staticmethod
    def nint_enhancement():
        top = tk.Toplevel()
        nint_params = Nint(top)

    @staticmethod
    def gauss_enhancement():
        top = tk.Toplevel()
        gauss_params = Gauss(top)

    @staticmethod
    def root_enhancement():
        top = tk.Toplevel()
        root_params = Root(top)


class Nint:

    def __init__(self, master):
        self.master = master
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(expand=True, fill="both")

        self.factor = tk.IntVar()
        self.factor.set(5)
        self.t = tk.IntVar()
        self.t.set(0.7)

        self.factor_label = tk.Label(self.main_frame, text="Factor", width=20)
        self.factor_label.grid(row=0, column=0)

        self.factor_entry = tk.Entry(self.main_frame, textvariable=self.factor, width=20)
        self.factor_entry.grid(row=0, column=1)

        self.t_label = tk.Label(self.main_frame, text=" t ", width=20)
        self.t_label.grid(row=1, column=0)

        self.t_entry = tk.Entry(self.main_frame, textvariable=self.t, width=20)
        self.t_entry.grid(row=1, column=1)

        self.set_button = tk.Button(self.main_frame, text=" Set Values ", command=self.set_values)
        self.set_button.grid(row=2, column=0)

        self.close_button = tk.Button(self.main_frame, text="Close", command=self.close_window)
        self.close_button.grid(row=2, column=1)

    def set_values(self):
        factor = self.factor_entry.get()
        self.factor.set(factor)
        t = self.t_entry.get()
        self.t.set(t)
        test.preview_image = ce.contrast_enhancement_nint_multi(rgb2gray(test.original_image),
                                                                [self.t.get()],
                                                                [self.factor.get()])[0]
        test.n_im_axes.imshow(test.preview_image, cmap='gray')

    def close_window(self):
        self.master.destroy()


class Gauss:

    def __init__(self, master):
        self.master = master
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(expand=True, fill="both")

        self.height = tk.DoubleVar()
        self.height.set(1)
        self.image_mean = tk.DoubleVar()
        self.image_mean.set(0.5)
        self.image_std = tk.DoubleVar()
        self.image_std.set(0.1)

        self.height_label = tk.Label(self.main_frame, text="Height", width=20)
        self.height_label.grid(row=0, column=0)

        self.height_entry = tk.Entry(self.main_frame, textvariable=self.height, width=20)
        self.height_entry.grid(row=0, column=1)

        self.mean_label = tk.Label(self.main_frame, text=" Mean ", width=20)
        self.mean_label.grid(row=1, column=0)

        self.mean_entry = tk.Entry(self.main_frame, textvariable=self.image_mean, width=20)
        self.mean_entry.grid(row=1, column=1)

        self.std_label = tk.Label(self.main_frame, text=" Std ", width=20)
        self.std_label.grid(row=2, column=0)

        self.std_entry = tk.Entry(self.main_frame, textvariable=self.image_std, width=20)
        self.std_entry.grid(row=2, column=1)

        self.set_button = tk.Button(self.main_frame, text=" Set Values ", command=self.set_values)
        self.set_button.grid(row=3, column=0)

        self.close_button = tk.Button(self.main_frame, text="Close", command=self.close_window)
        self.close_button.grid(row=3, column=1)

    def set_values(self):
        height = self.height_entry.get()
        self.height.set(height)

        mean = self.mean_entry.get()
        self.image_mean.set(mean)

        std = self.std_entry.get()
        self.image_std.set(std)

        test.preview_image = ce.gaussian_ce(rgb2gray(test.original_image),
                                            self.height.get(),
                                            self.image_mean.get(),
                                            self.image_std.get())
        test.n_im_axes.imshow(test.preview_image, cmap='gray')

    def close_window(self):
        self.master.destroy()


class Root:

    def __init__(self, master):
        self.master = master
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(expand=True, fill="both")

        self.root = tk.DoubleVar()
        self.root.set(2)

        self.times_label = tk.Label(self.main_frame, text="Root", width=10)
        self.times_label.grid(row=0, column=0)

        self.times_entry = tk.Entry(self.main_frame, textvariable=self.root, width=5)
        self.times_entry.grid(row=0, column=1)

        self.set_button = tk.Button(self.main_frame, text="  Set  ", command=self.set_values)
        self.set_button.grid(row=1, column=0)

        self.close_button = tk.Button(self.main_frame, text="Close", command=self.close_window)
        self.close_button.grid(row=1, column=1)

    def set_values(self):
        self.root.set(self.times_entry.get())
        root = self.root.get()
        test.preview_image = ce.root_ce(rgb2gray(test.original_image), root)
        test.n_im_axes.imshow(test.preview_image, cmap='gray')

    def close_window(self):
        self.master.destroy()


test = Previewer()

test.mainloop()
