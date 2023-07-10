import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from skimage.io import imread, imsave


class Window(tk.Tk):
    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        self.configure(takefocus=False)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        hs = self.winfo_screenheight() - 75
        ws = self.winfo_screenwidth() - 10
        self.minsize(width=ws, height=hs)
        self.geometry("{}x{}+0+0".format(ws, hs))

        self.fig_frame = tk.Frame(self, height=720, width=1080, takefocus=False)
        self.fig_frame.grid(row=0, column=0)
        self.fig_frame.columnconfigure(0, weight=1)
        self.fig_frame.rowconfigure(0, weight=1)

        self.fig = plt.figure(figsize=(15.8, 7.8))
        self.fig_axes = self.fig.gca()

        self.buttons_frame = tk.Frame(self, takefocus=False)
        self.buttons_frame.grid(row=1, column=0)

        self.toolbar_frame = tk.Frame(self, takefocus=False)
        self.toolbar_frame.grid(row=2, column=0)

        self.canvas = FigureCanvasTkAgg(self.fig, self.fig_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=10)

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        for w in self.toolbar.winfo_children():
            w.configure(takefocus=False)

        self.click = []
        self.shift = None
        self.width = tk.IntVar()
        self.width.set(125)
        self.height = tk.IntVar()
        self.height.set(180)
        self.rect = []
        self.rectangle_x = Rectangle([0, 0], 0, 0)
        self.count1 = 0

        plt.ion()

        self.browse_button = tk.Button(self.buttons_frame, text=" Browse ", command=self.browse_img, takefocus=False)
        self.browse_button.grid(row=1, column=0)

        self.width_label = tk.Label(self.buttons_frame, text=" width ", takefocus=False)
        self.width_label.grid(row=1, column=1, padx=5, pady=2)

        self.width_entry = tk.Entry(self.buttons_frame, textvariable=self.width, takefocus=False)
        self.width_entry.grid(row=1, column=2, padx=5, pady=2)

        self.height_label = tk.Label(self.buttons_frame, text=" height ", takefocus=False)
        self.height_label.grid(row=1, column=3, padx=5, pady=2)

        self.height_entry = tk.Entry(self.buttons_frame, textvariable=self.height, takefocus=False)
        self.height_entry.grid(row=1, column=4, padx=5, pady=2)

        self.set_button = tk.Button(self.buttons_frame, text="    Set    ", command=self.set_dimensions,
                                    takefocus=False)
        self.set_button.grid(row=1, column=5, padx=5, pady=2)

        self.save_button = tk.Button(self.buttons_frame, text=" Save ", command=self.save, takefocus=False)
        self.save_button.grid(row=1, column=6, padx=5, pady=2)

        self.close_button = tk.Button(self.buttons_frame, text=" Close ", command=self.close, takefocus=False)
        self.close_button.grid(row=1, column=7, padx=5, pady=2)

        self.refresh_button = tk.Button(self.buttons_frame, text=" Refresh ", command=self.refresh, takefocus=False)
        self.refresh_button.grid(row=1, column=8, padx=5, pady=2)

        self.clear_rects_button = tk.Button(self.buttons_frame, text="Clear Rectangles", command=self.clear_rects,
                                            takefocus=False)
        self.clear_rects_button.grid(row=1, column=9)

    def browse_img(self):
        path = askopenfilename()
        self.image = imread(path)
        self.fig_axes.imshow(self.image)
        self.kid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.krid = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_key(self, event):
        if event.inaxes == self.fig_axes:
            if event.key is "shift":
                self.shift = "shift"

    def on_key_release(self, event):
        if self.shift is "shift":
            self.shift = None

    def on_click(self, event):
        if event.inaxes == self.fig_axes:
            if self.shift is "shift":
                width = self.width.get()
                height = self.height.get()
                coordinates = [int(event.xdata), int(event.ydata), width, height]
                self.click.append(coordinates)
                print(self.click)
                self.rectangle_x = self.fig_axes.add_patch(
                    Rectangle(coordinates, width, height, edgecolor='r', fill=None))
                self.rect.append(self.rectangle_x)
                # self.fig.canvas.draw_idle()

    def set_dimensions(self):
        width = self.width_entry.get()
        self.width.set(width)
        height = self.height_entry.get()
        self.height.set(height)

    def refresh(self):
        self.click = []
        self.rect = []
        self.image = []
        self.width.set(125)
        self.height.set(180)
        self.fig.canvas.mpl_disconnect(self.cid)
        plt.clf()
        self.fig_axes = self.fig.add_axes([0.125, 0.11, 0.775, 0.77])

    def save(self):
        path = asksaveasfilename()
        self.count1 += 1
        count2 = 1
        for xy in range(len(self.click)):
            col_begin = self.click[xy][0]
            col_end = col_begin + self.click[xy][2]
            row_begin = self.click[xy][1]
            row_end = row_begin + self.click[xy][3]
            sample = self.image[row_begin:row_end, col_begin:col_end, :]
            file_path = path + "batch_0{}_sample_0{}.jpeg".format(self.count1, count2)
            imsave(file_path, sample)
            count2 += 1

    def close(self):
        plt.close('all')
        self.destroy()

    def clear_rects(self):
        self.click = []
        self.rect = []
        self.width.set(self.width_entry.get())
        self.height.set(self.height_entry.get())
        self.fig_axes.clear()
        self.fig_axes.imshow(self.image)

    def focus_frame(self):
        self.fig_frame.focus_set()


new_window = Window()
new_window.mainloop()
