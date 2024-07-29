import tkinter as tk                # python 3
from tkinter import font as tkfont  # python 3
import os
import rasterio
import albumentations
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.nn import functional as F
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import *
import numpy as np
import math
import cv2
import re
from skimage import measure
from shapely.geometry import shape
from rasterio.features import shapes
import time
import sys
import model
import decoder


bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))

image_path = ''
model_path = ''
output_dir = ''

image_path_entry = ''
model_path_entry = ''
output_path_entry = ''

info_text1 = ''
info_text2 = ''
info_text3 = ''
info_text4 = ''
info_text5 = ''
info_text6 = ''
info_text7 = ''
info_text8 = ''
info_text9 = ''
info_text10 = ''
info_text11 = ''

canvas2 = ''


class SampleApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack()

        self.frames = {}
        for F in (StartPage, PageOne):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        canvas = tk.Canvas(self, width=800, height=600)
        self.pack()
        canvas.pack()

        # Display image
        bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
        path_to_bg = os.path.abspath(os.path.join(bundle_dir, 'Background.png'))
        img = tk.PhotoImage(file=path_to_bg)
        controller.img = img
        canvas.create_image((0, 0), image=img, anchor='nw')

        canvas.create_rectangle(50, 80, 750, 470, fill='white', outline='', stipple='gray75')

        # Add a text in canvas
        canvas.create_text((250, 100), text="Welcome to CITY's Building Detector!",
                           font=("Times New Roman", 18))
        info_text = "This is a deep learning based Building Detector.\n" \
                    "You need following data to run this detector:\n" \
                    "\n" \
                    "1) A Satellite image with approximately 1 meter per pixel resolution in geo-tiff format.\n" \
                    "2) A trained deep learning model in .hdf format.\n" \
                    "\n" \
                    "The detector will automatically generate polygons over the detected buildings and save the results" \
                    " as a single csv file. The output csv file can be loaded directly in GIS by providing geometry" \
                    " column as geometry attribute. The CRS for output csv file will be same as input geo-tiff file."
        canvas.create_text((400, 250), text=info_text,
                           font=("Times New Roman", 14), width=680)
        button = tk.Button(canvas, text="Continue",
                           command=lambda: controller.show_frame("PageOne"))
        button.place(x=650, y=420)


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        global canvas2
        canvas2 = tk.Canvas(self, width=800, height=600)
        canvas2.pack()
        entry_width = 60

        # Display image
        path_to_bg = os.path.abspath(os.path.join(bundle_dir, 'Background.png'))
        one = tk.PhotoImage(file=path_to_bg)
        controller.one = one
        canvas2.create_image((0, 0), image=one, anchor='nw')
        canvas2.create_rectangle(50, 80, 750, 470, fill='white', outline='', stipple='gray75')
        image_select_button = ttk.Button(
            canvas2,
            text='Select Image',
            command=select_tiff_file
        )
        global image_path_entry, model_path_entry, output_path_entry
        image_path_entry = Entry(canvas2, width=entry_width)
        image_fix_text = tk.Label(master=canvas2, text="Enter path or select input image:")

        model_select_button = ttk.Button(
            canvas2,
            text='Select Model',
            command=select_model_file
        )
        model_path_entry = Entry(canvas2, width=entry_width)
        model_fix_text = tk.Label(master=canvas2, text="Enter path or select model file:")

        output_select_button = ttk.Button(
            canvas2,
            text='Select a Directory',
            command=select_output_dir
        )
        output_path_entry = Entry(canvas2, width=entry_width)
        output_fix_text = tk.Label(master=canvas2, text="Enter path or select output path:")

        run_button = ttk.Button(
            canvas2,
            text='Run Detector',
            command=run_model
        )
        global info_text1, info_text2, info_text3, info_text4
        global info_text5, info_text6, info_text7, info_text8
        global info_text9, info_text10, info_text11

        info_text1 = Label(canvas2, text="")
        info_text2 = Label(canvas2, text="")
        info_text3 = Label(canvas2, text="")
        info_text4 = Label(canvas2, text="")
        info_text5 = Label(canvas2, text="")
        info_text6 = Label(canvas2, text="")
        info_text7 = Label(canvas2, text="")
        info_text8 = Label(canvas2, text="")
        info_text9 = Label(canvas2, text="")
        info_text10 = Label(canvas2, text="")
        info_text11 = Label(canvas2, text="")

        x_start = 80
        y_start = 100

        image_fix_text.place(x=x_start, y=y_start)
        image_path_entry.place(x=x_start + 190, y=y_start)
        image_select_button.place(x=x_start + 560, y=y_start)

        model_fix_text.place(x=x_start, y=y_start + 30)
        model_path_entry.place(x=x_start + 190, y=y_start + 30)
        model_select_button.place(x=x_start + 560, y=y_start + 30)

        output_fix_text.place(x=x_start, y=y_start + 60)
        output_path_entry.place(x=x_start + 190, y=y_start + 60)
        output_select_button.place(x=x_start + 560, y=y_start + 60)

        run_button.place(x=x_start + 300, y=y_start + 100)

        info_text1.place(x=x_start, y=y_start + 120)
        info_text2.place(x=x_start, y=y_start + 140)
        info_text3.place(x=x_start, y=y_start + 160)
        info_text4.place(x=x_start, y=y_start + 180)
        info_text5.place(x=x_start, y=y_start + 210)
        info_text6.place(x=x_start, y=y_start + 230)
        info_text7.place(x=x_start, y=y_start + 260)
        info_text8.place(x=x_start, y=y_start + 280)
        info_text9.place(x=x_start, y=y_start + 300)
        info_text10.place(x=x_start, y=y_start + 330)
        info_text11.place(x=x_start, y=y_start + 350)

        button = tk.Button(canvas2, text="Back",
                           command=lambda: controller.show_frame("StartPage"))
        button.place(x=650, y=420)


def select_tiff_file():
    filetypes = (
        ('tiff files', '*.tif'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    global image_path, image_path_entry
    image_path = filename
    image_path_entry.insert(END, image_path)


def select_model_file():
    filetypes = (
        ('h5 files', '*.h5'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir=r'/',
        filetypes=filetypes)
    global model_path, model_path_entry
    model_path = filename
    model_path_entry.insert(END, model_path)


def select_output_dir():
    filename = fd.askdirectory(initialdir=r'/')
    global output_dir, output_path_entry
    output_dir = filename
    output_path_entry.insert(END, output_dir)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albumentations.Lambda(image=preprocessing_fn),
    ]
    return albumentations.Compose(_transform)


def save_as_csv(out_path, contours):
    out_path = os.path.join(out_path, 'polygons.csv')
    fw = open(out_path, 'w')
    fw.write("filename,id,Geometry\n")
    for j, contour in enumerate(contours):
        polygon_str = re.sub(r"[\[\]]", '', ",".join(map(str, contour)))
        fw.write("%s,%d,\"POLYGON ((%s))\"\n" % ("2022_09_12", j, polygon_str))
    fw.close()


def assign_geocoords(contours, img_path, y_flip=True):
    # img_path = get_respond_img(imgs_dir,npy_file)
    dataset = rasterio.open(img_path)

    x_min_coord = dataset.bounds[0]  # left edge
    x_max_coord = dataset.bounds[2]  # right edge
    y_min_coord = dataset.bounds[1]  # bottom edge
    y_max_coord = dataset.bounds[3]  # top edge

    y_img_size, x_img_size = dataset.read(1).shape
    dataset.close()

    for it in range(len(contours)):
        contour = contours[it]

        x = (contour[:, 0])
        y = (contour[:, 1])
        if (y_flip):
            y = y_img_size - (y)

        geo_x = x_min_coord + (x / x_img_size) * (x_max_coord - x_min_coord)
        geo_y = y_min_coord + (y / y_img_size) * (y_max_coord - y_min_coord)

        contours[it] = np.vstack((geo_x, geo_y)).T

    return contours


def run_model():
    target_size = (512, 512)
    padding_pixels = (64, 64)
    padding_value = 0
    downsampling_factor = 1
    PIX_VALUE_MAX = 255
    PIX_VALUE_MAX_REQ = 255

    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = torch.device("cpu")

    global output_dir, image_path, model_path
    global info_text1, info_text2, info_text3, info_text4
    global info_text5, info_text6, info_text7, info_text8
    global info_text9, info_text10, info_text11
    global canvas2

    output_dir = r'{}'.format(output_dir)
    image_path = r'{}'.format(image_path)
    model_path = r'{}'.format(model_path)

    info_text1.configure(text="The following paths are provided:")
    info_text2.configure(text="Image Path: {}".format(image_path))
    info_text3.configure(text="Model Path: {}".format(model_path))
    info_text4.configure(text="Output Directory: {}".format(output_dir))
    info_text5.configure(text="Attempting Image and Model Load")
    canvas2.update()

    img_found = True
    try:
        img = rasterio.open(image_path)
    except:
        img_found = False

    model_found = True
    try:
        model = torch.load(model_path, map_location=DEVICE)
    except Exception as e:
        print(e)
        model_found = False

    if (img_found and model_found):
        info_text5.configure(text="Attempting Image and Model Load.....Done!")
        info_text6.configure(text="All Okay..!")
    else:
        info_text5.configure(text="Attempting Image and Model Load.....Error!")
        text = "Error Could Not Load: "
        if not img_found:
            text += "Image file "
        if not model_found:
            text += "Model file "
        text += ",process stopped"
        info_text6.configure(text=text)
        assert True

    canvas2.update()

    if img_found and model_found:
        file_name = os.path.split(image_path)[-1].split('.')[0]
        model.eval()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_path = os.path.join(output_dir, file_name + "_preds.npy")
        img = np.transpose(img.read(), (1, 2, 0))

        # Define k_x, k_y to define 'useful' portion, since we are taking patches with overlapping area.
        k_y = target_size[0] - 2 * padding_pixels[0]
        k_x = target_size[1] - 2 * padding_pixels[1]

        # First padding: To make divisible by k
        cols = (math.ceil(img.shape[0] / k_y))
        rows = (math.ceil(img.shape[1] / k_x))

        pad_bottom = cols * k_y - img.shape[0]  # pixels to add in y direction
        pad_right = rows * k_x - img.shape[1]  # pixels to add in x direction
        if pad_bottom > 0 or pad_right > 0:
            img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=padding_value)
        output_image = np.zeros((int(img.shape[0] * downsampling_factor), int(img.shape[1] * downsampling_factor)),
                                dtype=np.uint8) * 255
        # Second Padding: To add boundary padding pixels
        img = cv2.copyMakeBorder(img, padding_pixels[0], padding_pixels[0], padding_pixels[1],
                                 padding_pixels[1], cv2.BORDER_CONSTANT, value=padding_value)

        # Load pre-processing function
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        preprocessing = get_preprocessing(preprocessing_fn)

        total_patches = rows * cols
        info_text7.configure(text="Total {} patches for the given image {}".format(rows * cols, file_name))
        canvas2.update()
        time_start = time.time()
        for y_idx in range(cols):
            y1 = y_idx * k_y + padding_pixels[0]
            y2 = y1 + k_y
            for x_idx in range(rows):
                x1 = x_idx * k_x + padding_pixels[1]
                x2 = x1 + k_x
                patch_number = y_idx * rows + x_idx + 1
                img_crop = img[y1 - padding_pixels[0]: y2 + padding_pixels[0],
                           x1 - padding_pixels[1]: x2 + padding_pixels[1]]
                img_crop = ((img_crop / PIX_VALUE_MAX) * (PIX_VALUE_MAX_REQ)).astype(np.uint8)
                sample = preprocessing(image=img_crop)
                image = cv2.resize(sample['image'],
                                   (int(downsampling_factor * target_size[0]),
                                    int(downsampling_factor * target_size[1])),
                                   interpolation=cv2.INTER_AREA)
                x_tensor = torch.Tensor(image).permute(2, 0, 1).to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                    pred_mask, _ = model(x_tensor)
                pr_mask = pred_mask.squeeze()
                pr_mask = pr_mask.detach().squeeze().cpu().numpy().round()

                patch = pr_mask[int(downsampling_factor * padding_pixels[0]): int(
                    downsampling_factor * (target_size[0] - padding_pixels[0])),
                        int(downsampling_factor * padding_pixels[1]): int(
                            downsampling_factor * (target_size[1] - padding_pixels[1]))]
                output_image[int(downsampling_factor * (y_idx * k_y)): int(downsampling_factor * (y_idx * k_y + k_y)),
                int(downsampling_factor * (x_idx * k_x)): int(downsampling_factor * (x_idx * k_x + k_x))] = patch
                time_taken = time.time() - time_start
                time_left = (total_patches - patch_number) * (time_taken/patch_number)
                info_text8.configure(text="Patch {} of {}, eta:(HH:MM:SS) {}".format(patch_number, total_patches,
                                                                           time.strftime("%H:%M:%S",
                                                                                         time.gmtime(time_left))))
                canvas2.update()

        info_text8.configure(text="Patch {} of {}, eta:(HH:MM:SS) {}...Done!".format(patch_number, total_patches,
                                                                           time.strftime("%H:%M:%S",
                                                                                         time.gmtime(time_left))))
        canvas2.update()
        output_image = output_image[:output_image.shape[0] - int(downsampling_factor * pad_bottom),
                       :output_image.shape[1] - int(downsampling_factor * pad_right)]
        np.save(save_path, output_image)
        info_text9.configure(text="Model Prediction Completed in {} seconds! npy file saved at: {}".format(time_taken,
                                                                                                           save_path))
        info_text10.configure(text="Starting polygonizing and geo-referencing...")
        canvas2.update()
        img = np.load(save_path)
        labels = measure.label(img, connectivity=2, background=0).astype('uint16')
        polygon_gen = shapes(labels, labels > 0)
        geoms_np = []
        ct = 0
        for polygon, value in polygon_gen:
            ct = ct + 1
            p = shape(polygon)
            if p.area >= 0:
                # p = p.simplify(tolerance=0.5)
                try:
                    p = np.array(p.boundary.xy, dtype='int32').T
                except:
                    p = np.array(p.boundary[0].xy, dtype='int32').T
                geoms_np.append(p)
        geo_poly = (assign_geocoords(geoms_np, image_path))
        save_as_csv(output_dir, geo_poly)
        info_text10.configure(text="Starting polygonizing and geo-referencing...Done!")
        info_text11.configure(text="Process Completed! File saved at: {}".format(os.path.join(output_dir,
                                                                                              'polygons.csv')))
        canvas2.update()


if __name__ == "__main__":
    window = SampleApp()
    path_to_ico = os.path.abspath(os.path.join(bundle_dir, "App Logo.ico"))
    window.iconbitmap(path_to_ico)
    window.title("CITY's Building Detector")
    window.geometry("800x600")
    window.resizable(width=False, height=False)
    window.mainloop()