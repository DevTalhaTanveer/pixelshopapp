from flask import render_template, Blueprint, request, jsonify
from PIL import Image, ImageEnhance, ImageFilter, ImageChops
import sys
import os
import numpy as np
from io import BytesIO
import base64
import cv2
from ..data import temperature_chart
from skimage import exposure
from cv2 import dnn_superres

global edited_image, to_save, real_image, width_img  # Global variables to hold image data
global hist_r, hist_g, hist_b, adjust_r_curve, adjust_g_curve, adjust_b_curve
global entry_1a, entry_1b, entry_2a, entry_2b, entry_3a, entry_3b
global slider_1a, slider_1b, slider_2a, slider_2b, slider_3a, slider_3b
global sweat_ranges 
sweat_ranges = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
# sweat_ranges = [[0, 255, 0, 255], [0, 255, 0, 255], [0, 255, 0, 255],[0,255,0,255]]
model_filename = "ESPCN_x4.pb"
model_path = os.path.join(os.path.dirname(__file__), model_filename)







    

edit_route = Blueprint("edit_route", __name__, template_folder='templates')

@edit_route.route('/resize', methods=['POST'])
def resize():
    response = request.json['response'].split(',')[1]
    measurement = request.json['measurement']
  
    width = measurement['widthValue']
    height = measurement['heightValue']
    nparray = np.frombuffer(base64.b64decode(response), np.uint8)
    img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
    response = cv2.resize(img, (int(width), int(height)), interpolation = cv2.INTER_LANCZOS4)
    res = base64.b64encode(cv2.imencode('.jpeg', response)[1])
    return jsonify({"res": str(res)})


def adjust_brightness(image, factor):
    """
    Adjusts the brightness of an image by multiplying each pixel value by a factor.

    Args:
        image (numpy.ndarray): The input image (BGR format).
        factor (float): The brightness adjustment factor. Use values > 1 to increase brightness,
                       values < 1 to decrease brightness.

    Returns:
        numpy.ndarray: The adjusted image.
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted_image




@edit_route.route('/others/brightness', methods=['POST'])
def brightness():
    response = request.json['response']
    factorial = request.json['factorial']
    print(factorial)
    basedir = base64.b64decode(response)
    basedir = BytesIO(basedir)
    im = Image.open(basedir)
    factor = factorial
    im_output = adjust_brightness(np.array(im), factor)
    im_output = Image.fromarray(im_output)
    buffered = BytesIO()
    
    rgb_im = im_output.convert('RGB')
    rgb_im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
  
    return jsonify({"res": str(res)})


@edit_route.route('/others/contrast', methods=['POST'])
def contrast():
    response = request.json['response']
    factorial = request.json['factorial']
    basedir = base64.b64decode(response)
    basedir = BytesIO(basedir)
    im = Image.open(basedir)
    factor = factorial
    enhancer = ImageEnhance.Contrast(im)
    im_output = enhancer.enhance(factor)
    buffered = BytesIO()
    rgb_im = im_output.convert('RGB')
    rgb_im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})


@edit_route.route('/others/sharpness', methods=['POST'])
def sharpness():
    response = request.json['response']
    factorial = request.json['factorial']
    basedir = base64.b64decode(response)
    basedir = BytesIO(basedir)
    im = Image.open(basedir)
    factor = factorial
    print(factor)
    enhancer = ImageEnhance.Sharpness(im)
    im_output = enhancer.enhance(factor)
    buffered = BytesIO()
    rgb_im = im_output.convert('RGB')
    rgb_im.save(buffered, format="JPEG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})


@edit_route.route('/color/saturation', methods=['POST'])
def saturation():
    response = request.json['response']
    factorial = request.json['factorial']
    basedir = base64.b64decode(response)
    basedir = BytesIO(basedir)
    im = Image.open(basedir)
    data = np.array(im)
    let = data[:1, :1, :]
    factor = factorial
    enhancer = ImageEnhance.Color(im)
    im_output = enhancer.enhance(factor)
    buffered = BytesIO()
    rgb_im = im_output.convert('RGB')
    rgb_im.save(buffered, format="JPEG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

@edit_route.route('/color/exposure', methods=['POST'])
def adjust_exposure():
    response = request.json['response']
    factorial = request.json['factorial']
    basedir = base64.b64decode(response)
    basedir = BytesIO(basedir)
    im = Image.open(basedir)
    factor = factorial
    im_output = exposure.adjust_gamma(np.array(im), factor)
    im_output = Image.fromarray(im_output)
    buffered = BytesIO()
    rgb_im = im_output.convert('RGB')
    rgb_im.save(buffered, format="JPEG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

def getRealImageAsPillow(real_image):
    b, g, r = cv2.split(real_image)
    img = Image.fromarray(np.dstack((r, g, b)))
    return img

def blend_images(img, img2, scale=0.7):
    img3 = Image.blend(img, img2, scale)
    return img3

@edit_route.route('/color/filter', methods=['POST'])
def adjust_filter():
    response = request.json['response']
    factorial = request.json['factorial']
    basedir = base64.b64decode(response)
    basedir = BytesIO(basedir)
    im = Image.open(basedir)
    factor = factorial
    im_output = getRealImageAsPillow(np.array(im))
    im_output = blend_images(im, im_output, 1 - float(factor))
    buffered = BytesIO()
    rgb_im = im_output.convert('RGB')
    rgb_im.save(buffered, format="JPEG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

def createImageHistogram(image):
    # Split the image into R, G, B channels
    b, g, r = cv2.split(image)

    # Create histograms for the adjusted channels
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    # Convert histograms to NumPy arrays
    hist_r = np.squeeze(hist_r)
    hist_g = np.squeeze(hist_g)
    hist_b = np.squeeze(hist_b)

    hist_r = (hist_r / max(hist_r)) * 255
    hist_g = (hist_g / max(hist_g)) * 255
    hist_b = (hist_b / max(hist_b)) * 255

    return hist_r, hist_g, hist_b

# def calculate_sweat_ranges(hist, num_ranges=4):
#   """
#   This function calculates four sweat ranges based on the given histogram.

#   Args:
#     hist: A list of integers representing the histogram values.
#     num_ranges: The number of sweat ranges to calculate (default: 4).

#   Returns:
#     A list of lists, where each sublist represents a sweat range.
#   """

#   sweat_ranges = []

#   # Calculate percentiles to divide the data into equal-sized ranges
#   percentiles = np.percentile(hist, np.linspace(0, 100, num_ranges + 1))

#   # Define sweat ranges based on percentiles
#   for i in range(1, num_ranges + 1):
#     min_val = percentiles[i - 1]
#     max_val = percentiles[i]
#     sweat_ranges.append([min_val, max_val, min_val, max_val])

#   return sweat_ranges
def autoAdjust(hist,ind ,t=True):
    global entry_1a, entry_1b, entry_2a, entry_2b, entry_3a, entry_3b
    global slider_1a, slider_1b, slider_2a, slider_2b, slider_3a, slider_3b
    global sweat_ranges
    
    f = 0
    b = 0
    if t:
        for i_ in range(2):
            for i in range(f, len(hist) - 1):
                if int(hist[i]) <= 2 or int(hist[i + 1]) <= 2:
                    f += 1
                else:
                    break

            for i in range(len(hist) - 2 - b, 0, -1):
                if int(hist[i]) <= 5 or int(hist[i + 1]) <= 5:
                    b += 1
                else:
                    break

            b += 1
            f += 1

    b = 255 - b
    print("f, b", f, b)
    
    if ind == 1:
        entry_1a = f
        entry_1b = b
    elif ind == 2:
        entry_2a = f
        entry_2b = b
    elif ind == 3:
        entry_3a = f
        entry_3b = b

    print("Entry",entry_1a, entry_1b)

    # ea.delete(0, 'end')
    # eb.delete(0, 'end')
    # ea.insert(0, str(f))
    # eb.insert(0, str(b))



    # sa.set(f)
    # sb.set(b)

    if t:
        if ind ==1: 
            # 24-238*0.35
            # sweat_ranges[0][0] = f - int(f*0.35) if f - int(f*0.35) > 0 else 0
            # sweat_ranges[0][1] = f + int(f*0.35)
            # sweat_ranges[0][2] = b - int((255-b)*0.35)
            # sweat_ranges[0][3] = b + int((255-b)*0.35) if b + int((255-b)*0.35) < 256 else 255
            sweat_ranges[0][0] = f
            sweat_ranges[0][1] = f
            sweat_ranges[0][2] =f
            sweat_ranges[0][3] = f
            sweat_ranges[1][0] = f+20
            sweat_ranges[1][1] = f+20
            sweat_ranges[1][2] = f+20
            sweat_ranges[1][3] = f+20
            sweat_ranges[2][0] = b
            sweat_ranges[2][1] = b
            sweat_ranges[2][2] = b
            sweat_ranges[2][3] = b
            sweat_ranges[3][0] = b-20
            sweat_ranges[3][1] = b-20
            sweat_ranges[3][2] = b-20
            sweat_ranges[3][3] = b-20

        elif ind ==2:
            # sweat_ranges[1][0] = f - int(f*0.35) if f - int(f*0.35) > 0 else 0
            # sweat_ranges[1][1] = f + int(f*0.35)
            # sweat_ranges[1][2] = b - int((255-b)*0.35)
            # sweat_ranges[1][3] = b + int((255-b)*0.35) if b + int((255-b)*0.35) < 256 else 255
            sweat_ranges[0][0] = f
            sweat_ranges[0][1] = f
            sweat_ranges[0][2] =f
            sweat_ranges[0][3] = f
            sweat_ranges[1][0] = f+20
            sweat_ranges[1][1] = f+20
            sweat_ranges[1][2] = f+20
            sweat_ranges[1][3] = f+20
            sweat_ranges[2][0] = b
            sweat_ranges[2][1] = b
            sweat_ranges[2][2] = b
            sweat_ranges[2][3] = b
            sweat_ranges[3][0] = b-20
            sweat_ranges[3][1] = b-20
            sweat_ranges[3][2] = b-20
            sweat_ranges[3][3] = b-20



        elif ind ==3:
            # sweat_ranges[2][0] = f - int(f*0.35) if f - int(f*0.35) > 0 else 0
            # sweat_ranges[2][1] = f + int(f*0.35)
            # sweat_ranges[2][2] = b - int((255-b)*0.35)
            # sweat_ranges[2][3] = b + int((255-b)*0.35) if b + int((255-b)*0.35) < 256 else 255
            sweat_ranges[0][0] = f
            sweat_ranges[0][1] = f
            sweat_ranges[0][2] =f
            sweat_ranges[0][3] = f
            sweat_ranges[1][0] = f+20
            sweat_ranges[1][1] = f+20
            sweat_ranges[1][2] = f+20
            sweat_ranges[1][3] = f+20
            sweat_ranges[2][0] = b
            sweat_ranges[2][1] = b
            sweat_ranges[2][2] = b
            sweat_ranges[2][3] = b
            sweat_ranges[3][0] = b-20
            sweat_ranges[3][1] = b-20
            sweat_ranges[3][2] = b-20
            sweat_ranges[3][3] = b-20


    print(f"the 0------{ sweat_ranges}")
   


def adjust_channel_curve(channel, curve_points):
    # Create a lookup table for the curve adjustment
    lut = np.arange(256, dtype=np.uint8)
    lut = np.interp(lut, curve_points[:, 0], curve_points[:, 1])

    # Apply the lookup table to the channel
    adjusted_channel = cv2.LUT(channel, lut)

    return adjusted_channel.astype(dtype=np.uint8)


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_LANCZOS4):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r),height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

# def update_histograms(i,entry_1a,entry_1b,entry_2a,entry_2b,entry_3a,entry_3b,slider_1a,slider_1b,slider_2a,slider_2b,slider_3a,slider_3b):
#     if entry_1a.get() == "" or entry_2a.get() == "" or entry_3a.get() == "" or entry_1b.get() == "" or entry_2b.get() == "" or entry_3b.get() == "":
#         return
#     print(i)
#     # print(entry_1a.get(), entry_1b.get(), entry_2a.get(), entry_2b.get(), entry_3a.get(), entry_3b.get())

#     try:
#         # checkEntries()
#         if i == "Green Channel":
#             # slider_hue_g_2.place(relx=0.1, rely=0.15, relwidth=0.8, relheight=0.03)
#             slider_2a.place(relx=0.08, rely=0.72, relheight=0.04, relwidth=0.42)
#             slider_2b.place(relx=0.5, rely=0.72, relheight=0.04, relwidth=0.42)

#             entry_2a.place(relx=0.2, rely=0.67, relheight=0.04, relwidth=0.2)
#             entry_2b.place(relx=0.6, rely=0.67, relheight=0.04, relwidth=0.2)

#             entry_1a.place_forget()
#             entry_1b.place_forget()
#             entry_3a.place_forget()
#             entry_3b.place_forget()

#             slider_1a.place_forget()
#             slider_1b.place_forget()
#             slider_3a.place_forget()
#             slider_3b.place_forget()

#             # Update the curve points based on slider values
#             r_curve_points = np.array([[int(entry_2a.get()), 0], [int(entry_2b.get()), 255]], dtype=np.uint8)
#             print(r_curve_points)
#             ax1.clear()
#             ax1.bar(range(256), hist_g, color='green', alpha=0.7, edgecolor="green")
#             ax1.plot(r_curve_points[:, 0], r_curve_points[:, 1], 'k-', label='Curve')
#             ax1.set_title('Green Channel')
#             ax1.set_xlim([0, 256])
#             ax1.set_ylim([0, 256])

#             ax1.axvline(x=sweat_ranges[1][0].get(), label='axvline - full height', ymin=0, ymax=255)
#             ax1.axvline(x=sweat_ranges[1][1].get(), label='axvline - full height', ymin=0, ymax=255)
#             ax1.axvline(x=sweat_ranges[1][2].get(), label='axvline - full height', ymin=0, ymax=255)
#             ax1.axvline(x=sweat_ranges[1][3].get(), label='axvline - full height', ymin=0, ymax=255)

#         elif i == "Blue Channel":
#             slider_hue_b_3.place(relx=0.1, rely=0.15, relwidth=0.8, relheight=0.03)
#             slider_3a.place(relx=0.08, rely=0.72, relheight=0.04, relwidth=0.42)
#             slider_3b.place(relx=0.5, rely=0.72, relheight=0.04, relwidth=0.42)

#             entry_3a.place(relx=0.2, rely=0.67, relheight=0.04, relwidth=0.2)
#             entry_3b.place(relx=0.6, rely=0.67, relheight=0.04, relwidth=0.2)

#             entry_1a.place_forget()
#             entry_1b.place_forget()
#             entry_2a.place_forget()
#             entry_2b.place_forget()

#             slider_2a.place_forget()
#             slider_2b.place_forget()
#             slider_1a.place_forget()
#             slider_1b.place_forget()

#             # Update the curve points based on slider values
#             r_curve_points = np.array([[entry_3a.get(), 0], [entry_3b.get(), 255]], dtype=np.uint8)
#             ax1.clear()
#             ax1.bar(range(256), hist_b, color='blue', alpha=0.7, edgecolor="blue")
#             ax1.plot(r_curve_points[:, 0], r_curve_points[:, 1], 'k-', label='Curve')
#             ax1.set_title('Blue Channel')
#             ax1.set_xlim([0, 256])
#             ax1.set_ylim([0, 256])

#             ax1.axvline(x=sweat_ranges[2][0].get(), label='axvline - full height', ymin=0, ymax=255)
#             ax1.axvline(x=sweat_ranges[2][1].get(), label='axvline - full height', ymin=0, ymax=255)
#             ax1.axvline(x=sweat_ranges[2][2].get(), label='axvline - full height', ymin=0, ymax=255)
#             ax1.axvline(x=sweat_ranges[2][3].get(), label='axvline - full height', ymin=0, ymax=255)

#         elif i == "Red Channel" or i == None:
#             slider_hue_r_1.place(relx=0.1, rely=0.15, relwidth=0.8, relheight=0.03)
#             slider_1a.place(relx=0.08, rely=0.72, relheight=0.04, relwidth=0.42)
#             slider_1b.place(relx=0.5, rely=0.72, relheight=0.04, relwidth=0.42)

#             entry_1a.place(relx=0.2, rely=0.67, relheight=0.04, relwidth=0.2)
#             entry_1b.place(relx=0.6, rely=0.67, relheight=0.04, relwidth=0.2)

#             entry_2a.place_forget()
#             entry_2b.place_forget()
#             entry_3a.place_forget()
#             entry_3b.place_forget()

#             slider_2a.place_forget()
#             slider_2b.place_forget()
#             slider_3a.place_forget()
#             slider_3b.place_forget()

#             # Update the curve points based on slider values
#             r_curve_points = np.array([[int(entry_1a.get()), 0], [int(entry_1b.get()), 255]], dtype=np.uint8)
#             ax1.clear()
#             ax1.bar(range(256), hist_r, color='red', alpha=0.7, edgecolor="red")
#             ax1.plot(r_curve_points[:, 0], r_curve_points[:, 1], 'k-', label='Curve')
#             ax1.set_title('Red Channel')
#             ax1.set_xlim([0, 256])
#             ax1.set_ylim([0, 256])

#             ax1.axvline(x=sweat_ranges[0][0].get(), label='axvline - full height', ymin=0, ymax=255)
#             ax1.axvline(x=sweat_ranges[0][1].get(), label='axvline - full height', ymin=0, ymax=255)
#             ax1.axvline(x=sweat_ranges[0][2].get(), label='axvline - full height', ymin=0, ymax=255)
#             ax1.axvline(x=sweat_ranges[0][3].get(), label='axvline - full height', ymin=0, ymax=255)

#         ax1.grid(True, linestyle='--', linewidth=0.5)
#         GraphSliderRunner(i)
#     except:
#         return



def applyModel(I, idx = 0):
    """Apply the model to the image I"""
    global edited_image, real_image, hist_r, hist_g, hist_b, width_img
    global entry_1a, entry_1b, entry_2a, entry_2b, entry_3a, entry_3b
    global slider_1a, slider_1b, slider_2a, slider_2b, slider_3a, slider_3b
    global sweat_ranges 
    global adjust_r_curve, adjust_g_curve, adjust_b_curve
    try:
        # Apply the correction model
        h, w = I.shape[:2]
        print(h, w)
        width_img = w
        if h >= w:
            I = ResizeWithAspectRatio(I, height=1024)
        else:
            I = ResizeWithAspectRatio(I, width=1024)

        real_image = I

        hist_r, hist_g, hist_b = createImageHistogram(I)
        b, g, r = cv2.split(I)
        if idx == 0:
            autoAdjust(hist_r,1,t=True)
            print("Entry",entry_1a, entry_1b) 
            adjust_r_curve = adjust_channel_curve(r,
                                                np.array([[entry_1a, 0], [entry_1b, 255]], dtype=np.uint8))
            autoAdjust(hist_g,2,t=False)
            adjust_g_curve = adjust_channel_curve(g,
                                                    np.array([[entry_2a, 0], [entry_2b, 255]], dtype=np.uint8))
            autoAdjust(hist_b,3,t=False)
            adjust_b_curve = adjust_channel_curve(b,
                                                    np.array([[entry_3a, 0], [entry_3b, 255]], dtype=np.uint8))

        elif idx == 1:
            autoAdjust(hist_r,1,t=False)
            adjust_r_curve = adjust_channel_curve(r,
                                                np.array([[entry_1a, 0], [entry_1b, 255]], dtype=np.uint8))
            autoAdjust(hist_g,2,t=True)
            adjust_g_curve = adjust_channel_curve(g,
                                                    np.array([[entry_2a, 0], [entry_2b, 255]], dtype=np.uint8))
            autoAdjust(hist_b,3,t=False)
            adjust_b_curve = adjust_channel_curve(b,
                                                    np.array([[entry_3a, 0], [entry_3b, 255]], dtype=np.uint8))
        elif idx == 2:
            autoAdjust(hist_r,1,t=False)
            adjust_r_curve = adjust_channel_curve(r,
                                                np.array([[entry_1a, 0], [entry_1b, 255]], dtype=np.uint8))
            autoAdjust(hist_g,2,t=False)
            adjust_g_curve = adjust_channel_curve(g,
                                                    np.array([[entry_2a, 0], [entry_2b, 255]], dtype=np.uint8))
            autoAdjust(hist_b,3,t=True)
            adjust_b_curve = adjust_channel_curve(b,
                                                    np.array([[entry_3a, 0], [entry_3b, 255]], dtype=np.uint8))
        print("adjust_r_curve",adjust_r_curve)
        print("adjust_g_curve",adjust_g_curve)
        print("adjust_b_curve",adjust_b_curve)

        edited_image = np.dstack((adjust_r_curve, adjust_g_curve, adjust_b_curve))
        

    except:
        return



@edit_route.route('/color/red', methods=['POST'])
def adjust_green_channel():
    global edited_image, real_image, hist_r, hist_g, hist_b, width_img
    response = request.json['response']
    # entry_1a = request.json['entry_1a']
    # entry_1b = request.json['entry_1b']
    # slider_1a = request.json['slider_1a']
    # slider_1b = request.json['slider_1b']
    # sweat_ranges = request.json['sweat_ranges']
    print(sweat_ranges)
    nparr = np.frombuffer(base64.b64decode(response), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    applyModel(img)

    buffered = BytesIO()
    rgb_im = Image.fromarray(edited_image)
    rgb_im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res), "hist_r": hist_r.tolist(), "hist_g": hist_g.tolist(), "hist_b": hist_b.tolist(), "width_img": width_img, "entry_1a": entry_1a, "entry_1b": entry_1b, "Sweat_Range":sweat_ranges })

@edit_route.route('/color/green', methods=['POST'])
def adjust_blue_channel():
    global edited_image, real_image, hist_r, hist_g, hist_b, width_img
    global entry_2a, entry_2b
    response = request.json['response']
    # entry_2a = request.json['entry_2a']
    # entry_2b = request.json['entry_2b']
    # slider_2a = request.json['slider_2a']
    # slider_2b = request.json['slider_2b']
    # sweat_ranges = request.json['sweat_ranges']
    nparr = np.frombuffer(base64.b64decode(response), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    applyModel(img, idx = 1)

    buffered = BytesIO()
    rgb_im = Image.fromarray(edited_image)
    rgb_im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    # update_histograms('Green Channel',entry_1a,entry_1b,entry_2a,entry_2b,entry_3a,entry_3b,slider_1a,slider_1b,slider_2a,slider_2b,slider_3a,slider_3a,slider_3b)

   
    return jsonify({"res": str(res), "hist_r": hist_r.tolist(), "hist_g": hist_g.tolist(), "hist_b": hist_b.tolist(), "width_img": width_img, "entry_2a": entry_2a, "entry_2b": entry_2b, "Sweat_Range":sweat_ranges })

@edit_route.route('/color/blue', methods=['POST'])
def adjust_red_channel():
    global edited_image, real_image, hist_r, hist_g, hist_b, width_img
    response = request.json['response']
    nparr = np.frombuffer(base64.b64decode(response), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    applyModel(img, idx = 2)

    buffered = BytesIO()
    rgb_im = Image.fromarray(edited_image)
    rgb_im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res), "hist_r": hist_r.tolist(), "hist_g": hist_g.tolist(), "hist_b": hist_b.tolist(), "width_img": width_img, "entry_3a": entry_3a, "entry_3b": entry_3b,"Sweat_Range":sweat_ranges  })


@edit_route.route('/slider1/green', methods=['POST'])
def slider1_green():
    response = request.json['response']
    factor = request.json['factorial']
    print(factor)
    nparr = np.frombuffer(base64.b64decode(response), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    adjust_r_curve = adjust_channel_curve(r,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    adjust_g_curve = adjust_channel_curve(g,
                                            np.array([[factor, 0], [255, 255]], dtype=np.uint8))
    adjust_b_curve = adjust_channel_curve(b,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    print(adjust_g_curve)
    edit_image = np.dstack((adjust_r_curve, adjust_g_curve, adjust_b_curve))
    buffered = BytesIO()
    rgb_im = Image.fromarray(edit_image)
    rgb_im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})


@edit_route.route('/slider2/green', methods=['POST'])
def slider2_green():
    response = request.json['response']
    factor = request.json['factorial']
    print(factor)
    nparr = np.frombuffer(base64.b64decode(response), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    adjust_r_curve = adjust_channel_curve(r,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    adjust_g_curve = adjust_channel_curve(g,
                                            np.array([[0, 0], [factor, 255]], dtype=np.uint8))
    adjust_b_curve = adjust_channel_curve(b,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    print(adjust_g_curve)
    edit_image = np.dstack((adjust_r_curve, adjust_g_curve, adjust_b_curve))
    buffered = BytesIO()
    rgb_im = Image.fromarray(edit_image)
    rgb_im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

@edit_route.route('/slider1/blue', methods=['POST'])
def slider1_blue():
    response = request.json['response']
    factor = request.json['factorial']
    print(factor)
    nparr = np.frombuffer(base64.b64decode(response), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    adjust_r_curve = adjust_channel_curve(r,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    adjust_g_curve = adjust_channel_curve(g,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    adjust_b_curve = adjust_channel_curve(b,
                                            np.array([[factor, 0], [255, 255]], dtype=np.uint8))
    print(adjust_g_curve)
    edit_image = np.dstack((adjust_r_curve, adjust_g_curve, adjust_b_curve))
    buffered = BytesIO()
    rgb_im = Image.fromarray(edit_image)
    rgb_im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

@edit_route.route('/slider2/blue', methods=['POST'])
def slider2_blue():
    response = request.json['response']
    factor = request.json['factorial']
    print(factor)
    nparr = np.frombuffer(base64.b64decode(response), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    adjust_r_curve = adjust_channel_curve(r,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    adjust_g_curve = adjust_channel_curve(g,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    adjust_b_curve = adjust_channel_curve(b,
                                            np.array([[0, 0], [factor, 255]], dtype=np.uint8))
    print(adjust_g_curve)
    edit_image = np.dstack((adjust_r_curve, adjust_g_curve, adjust_b_curve))
    buffered = BytesIO()
    rgb_im = Image.fromarray(edit_image)
    rgb_im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

@edit_route.route('/slider1/red', methods=['POST'])
def slider1_red():
    response = request.json['response']
    factor = request.json['factorial']
    print(factor)
    nparr = np.frombuffer(base64.b64decode(response), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    adjust_r_curve = adjust_channel_curve(r,
                                            np.array([[factor, 0], [255, 255]], dtype=np.uint8))
    adjust_g_curve = adjust_channel_curve(g,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    adjust_b_curve = adjust_channel_curve(b,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    print(adjust_g_curve)
    edit_image = np.dstack((adjust_r_curve, adjust_g_curve, adjust_b_curve))
    buffered = BytesIO()
    rgb_im = Image.fromarray(edit_image)
    rgb_im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

@edit_route.route('/slider2/red', methods=['POST'])
def slider2_red():
    response = request.json['response']
    factor = request.json['factorial']
    print(factor)
    nparr = np.frombuffer(base64.b64decode(response), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    adjust_r_curve = adjust_channel_curve(r,
                                            np.array([[0, 0], [factor, 255]], dtype=np.uint8))
    adjust_g_curve = adjust_channel_curve(g,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    adjust_b_curve = adjust_channel_curve(b,
                                            np.array([[0, 0], [255, 255]], dtype=np.uint8))
    print(adjust_g_curve)
    edit_image = np.dstack((adjust_r_curve, adjust_g_curve, adjust_b_curve))
    buffered = BytesIO()
    rgb_im = Image.fromarray(edit_image)
    rgb_im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

@edit_route.route('/hue/green', methods=['POST'])
def adjust_hue_green():
    global edited_image, real_image, hist_r, hist_g, hist_b, width_img
    response = request.json['response']
    factor = request.json['factorial']
    basedur = base64.b64decode(response)
    basedur = BytesIO(basedur)
    im = Image.open(basedur)
    layer = Image.new('RGB', im.size, 'green')
    output = Image.blend(im, layer, factor)
    buffered = BytesIO()
    output.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

@edit_route.route('/hue/blue', methods=['POST'])
def adjust_hue_blue():
    global edited_image, real_image, hist_r, hist_g, hist_b, width_img
    response = request.json['response']
    factor = request.json['factorial']
    basedur = base64.b64decode(response)
    basedur = BytesIO(basedur)
    im = Image.open(basedur)
    layer = Image.new('RGB', im.size, 'blue')
    output = Image.blend(im, layer, factor)
    buffered = BytesIO()
    output.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

@edit_route.route('/hue/red', methods=['POST'])
def adjust_hue_red():
    global edited_image, real_image, hist_r, hist_g, hist_b, width_img
    response = request.json['response']
    factor = request.json['factorial']
    basedur = base64.b64decode(response)
    basedur = BytesIO(basedur)
    im = Image.open(basedur)
    layer = Image.new('RGB', im.size, 'red')
    output = Image.blend(im, layer, factor)
    buffered = BytesIO()
    output.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

    


    



@edit_route.route('/color/temperature', methods=['POST'])
def temperature():
    response = request.json['response']
    factorial = request.json['factorial']
    basedir = base64.b64decode(response)
    basedir = BytesIO(basedir)
    im = Image.open(basedir)
    im = im.convert('RGB')

    mode = im.mode
    red, blue, green = temperature_chart[factorial]  
    matrix = (red / 255, 0, 0, 0, 0, 
              green / 255, 0, 0, 0, 0, 
              blue / 255, 0, )
    im = im.convert('RGB',matrix=matrix)
    buffered = BytesIO()
    im.save(buffered, format="PNG",quality=100)
    res = base64.b64encode(buffered.getvalue())
    return jsonify({"res": str(res)})

@edit_route.route('/upscale', methods=['POST'])
def upscale_image():
    # Get the JSON data from the request
    response = request.json['response']
    basedir = base64.b64decode(response)
    basedir = BytesIO(basedir)
    im = Image.open(basedir)
    print(im)
    im_np = np.array(im)
    im_np = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("espcn", 4) # set the model by passing the value and the upsampling ratio
    result = sr.upsample(im_np) # upscale the input image
    _, buffer = cv2.imencode('.png', result)
    result_base64 = base64.b64encode(buffer).decode('utf-8')

    





    return jsonify({'res': result_base64})