"""Main module."""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
import skimage

import imageio
import tifffile

import os

def do_mli(
    img_filename, 
    save_pic = True,
    save_dir = "output",
    min_chord_length = 3.0,
    max_chord_length = 500.0,
    max_image_length = 3500,
):
    '''
    Given the filename of an H&E image, this 
    function will convert it to grayscale,
    equalize the histogram to correct poor contrast,
    give it a light blur, perform global thresholding,
    and then trace lines over top of it to
    measure mean chord lengths.
    
    Parameters
    ----------

    img_file: str
        The path to the image file
    
    save_dir: str (default: "chord_length_measurement_output")
        The subdirectory (under the parent directory of `img_file`)
        to store output images in
    
    save_pic: bool (default: True)
        Will save intermediate images
    
    min_chord_length: float (default: 3.0)
        Minimum length of chord to be considered
        a valid chord for measurement, in um
    
    max_chord_length: float (default: 500.0)
        Maximum length of chord, as for 
        `min_chord_length`
    
    max_image_length: int (default: 3500)
        Maximum length (in pixels) of either axis of
        the input image; images with an axis greater than
        this length will be rescaled down to fit
        for computational efficiency
    
    Returns
    -------  

    ret: mean of chord lengths in um
    
    '''

    # load the image and do the thresholding
    if ("/" in img_filename):
        img_path, img_filename = os.path.split(img_filename)
    img_ext = "." + img_filename.split(".")[-1]


    res, orig_img, img, global_otsu = make_thresholded_images(
        os.path.join(img_path, img_filename),
        max_image_length
    ) 
    
    # calculate chord spacing from image resolution
    lines_per_pix  = (2/50.0) * (1/res)
    pix_per_line   = int(1 / lines_per_pix)

    # make the chord image
    chords = make_chord_image(
        global_otsu, 
        pix_per_line
    )
    
    # measure the chords
    label_start = 5000
        
    chord_lengths, label_img = measure_chords(
        chords,
        pix_per_line,
        label_start
    )

    ret = []
    for i, val in enumerate(chord_lengths):
        x = val / res
        if (min_chord_length < x < max_chord_length):
            ret.append(round(x, 3))
        else:
            mask = label_img == (i + label_start)
            label_img[mask] = 0
    
    # Plot the resultant images
    figure_scale = 8
    if (orig_img.shape[1] < orig_img.shape[0]):
        figsize_x, figsize_y = (figure_scale * orig_img.shape[1] / orig_img.shape[0], figure_scale)
    else:
        figsize_x, figsize_y = (figure_scale, figure_scale * orig_img.shape[0] / orig_img.shape[1])
    
    fig, axs = plt.subplots(
        2,2,
        figsize=(figsize_x, figsize_y),
        gridspec_kw = {"wspace": 0.1, "hspace": 0.1}
    )
    
    axs[0,0].imshow(orig_img)
    axs[0,0].set_title("original")

    axs[0,1].imshow(img, cmap=plt.cm.gray)
    axs[0,1].set_title("gray-contrast-adjusted")

    axs[1,0].imshow(global_otsu, cmap=plt.cm.gray)
    axs[1,0].set_title("threshold-global-otsu")

    axs[1,1].imshow(label_img, cmap=plt.cm.terrain_r, vmin=0)
    axs[1,1].set_title("chords")

    for a in axs.ravel():
        a.set_axis_off()
    
    plt.suptitle(img_filename)
    
    # save the image
    if (save_pic):
        if not (os.path.exists(os.path.join(img_path, save_dir))):
            os.makedirs(os.path.join(img_path, save_dir))
        fig.savefig(os.path.join(
            img_path, save_dir, img_filename.split(".")[0] + "_output.png"
        ), dpi=150)
    
    # return the mean length
    return np.mean(ret)


def measure_chords(
    chord_img: np.ndarray, 
    pix_per_line: int, 
    label_start: int = 0
):
    '''
    Measures and labels chords if they do not 
    touch either edge of the image
    '''

    label = label_start
    label_img = np.zeros_like(chord_img, dtype=np.uint)
    chord_lengths = []
    idy = pix_per_line
    while (idy < chord_img.shape[0]):
        row = chord_img[idy, :]
        line_start = None
        
        for idx, val in enumerate(row):
            if (val and (line_start is None)): # started a new chord
                line_start = idx
            elif (not val and (line_start is not None)): # ended a chord
                if (line_start > 2): # chord does not touch the starting edge of the image
                    chord_lengths.append(idx - line_start) 
                    label_img[idy, line_start:idx] = label
                    label += 1
                line_start = None
        
        idy += pix_per_line
        if (line_start is not None): # ended a chord, but it touches the other edge of the image
            continue
        
    return chord_lengths, label_img

def make_thresholded_images(
    img_filename: str,
    max_image_length: int = 3500
):
    '''
    Load an H&E image from a file,
    determine its resolution, and generate
    intermediate contrast-adjusted
    and thresholded images, rescaling if
    necessary.

    Parameters
    ----------

    img_filename
        Path to the original H&E image.

    max_image_length
        Maximum length (in pixels) of either axis of
        the input image; images with an axis greater
        than this length will be rescaled down to fit
        for computational efficiency.

    Returns
    -------
    
    res: float
        How many pixels per um
        in the size-rescaled image

    orig_img: np.ndarray
        The original H&E image

    img: np.ndarray
        The contrast-adjusted, size-rescaled
        image.

    global_otsu: np.ndarray (np.bool)
        The thresholded image.

    '''

    # load the image from disk
    if ("/" in img_filename):
        img_path, img_filename = os.path.split(img_filename)
    img_ext = "." + img_filename.split(".")[-1]
    
    img = skimage.img_as_ubyte(
        imageio.v2.imread(img_path + "/" + img_filename)
    )
    
    # get the image metadata
    with tifffile.TiffFile(img_path + "/" + img_filename) as tif:
        for page in tif.pages:
            for tag in page.tags:
                if ("XResolution" in tag.name):
                    # XResolution is a tuple (numerator, denominator)
                    # and is usually given in pixels per cm, not um
                    res = tag.value[0] / tag.value[1]
                    res /= 1e4 #cm to um

    # copy for plotting
    orig_img = img.copy()
    
    # convert to grey
    img = skimage.color.rgb2gray(img)
    
    # correct black pixels (masked)
    img[img == 0] = np.mean(img[img > 0])
    
    # blur a little
    img = sp.ndimage.gaussian_filter(img, sigma=3)

    # if image is too big, scale it down
    scale_factor = 1
    max_input_img_axis_length = np.max(img.shape)
    if (max_input_img_axis_length > max_image_length):
        scale_factor = max_image_length / max_input_img_axis_length
        img = skimage.transform.rescale(
            img, 
            scale=scale_factor, 
            anti_aliasing=False
        )
        res *= scale_factor 
    
    # adjust for under-exposure
    img = skimage.exposure.equalize_adapthist(img)
    
    # otsu
    threshold_global_otsu = threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu

    return res, orig_img, img, global_otsu


def make_chord_image(
    img: np.ndarray,
    pix_per_line: int
):
    '''
    Generate an np.ndarray of split
    chords from an H&E image.

    Parameters
    ----------

    img
        The thresholded H&E image.

    pix_per_line
        How many pixels to space out
        horizontal lines (chords) by.
    
    Returns
    -------

    chords: np.ndarray (np.bool)
        An np.ndarray matching the shape
        of the rescaled, thresholded H&E image
        containing chords.

    '''

    chords = np.zeros_like(img, dtype=bool)
    i = pix_per_line
    while (i < chords.shape[0]):
        chords[i,:] = True
        i += pix_per_line
    chords = ~chords < img

    return chords