import numpy as np

from skimage.filters import threshold_otsu
import skimage

import imageio
import tifffile


def _get_tiff_resolution(
    img_filename: str,
):
    """
    Given the path to a tiff file, 
    read the metadata to determine
    the resolution of the image
    in pixels per um. 

    Parameters
    ----------

    img_filename
        The full path to the tiff image

    Returns
    -------

    res: float
        The number of pixels per um
        for the given image
    """

    # get the image metadata
    with tifffile.TiffFile(img_filename) as tif:
        for page in tif.pages:
            for tag in page.tags:
                if ("XResolution" in tag.name):
                    # XResolution is a tuple (numerator, denominator)
                    # and is usually given in pixels per cm, not um
                    res = tag.value[0] / tag.value[1]
                    if (res/1e4 < 0.01):
                        return res # already in um probably
                    else:
                        return res/1e4 # convert from cm to um
    if not res:
        raise ValueError("XResolution not found in tiff metadata.") 


def _make_thresholded_images(
    img_filename: str,
    max_image_length: int = 10000,
    greyscale: bool = True,
    channel_axis: int = None,
):
    """
    Load an H&E image from a file,
    determine its resolution, and generate
    intermediate contrast-adjusted
    and thresholded images, rescaling if
    necessary.

    Parameters
    ----------

    img_filename
        Path to the original H&E image.

    max_image_length: int (default: 10000)
        Maximum length (in pixels) of either axis of
        the input image; images with an axis greater
        than this length will be rescaled down to fit
        for computational efficiency.

    greyscale: bool (default: True)
        If True, `img` will be returned as grayscale.
        If False, `img` will be reshaped, but will maintain
        its color axis.

    channel_axis: int (default: None)
        The axis corresponding to color channels after loading
        with `imagio.v3.imread(f)`. If None, will determine
        based on axis shapes (smallest axis is channel axis).

    Returns
    -------
    
    res: float
        How many pixels per um
        in the size-rescaled image

    orig_img: np.ndarray
        The original H&E image

    img: np.ndarray
        The contrast-adjusted, size-rescaled image.

    global_otsu: np.ndarray (np.bool)
        The thresholded image.

    """

    # load the image from disk
    
    img = imageio.v3.imread(img_filename)
    
    if not (channel_axis):
        if (len(img.shape) < 3):
            raise ValueError(f"Image too small to contain channel axis! {img.shape}, {img_filename}")
        else:
            channel_axis = np.argmin(img.shape)

    #img = np.moveaxis(img, 0, -1)

    # get the image metadata
    res = _get_tiff_resolution(img_filename)

    # copy for plotting
    orig_img = img.copy()
    
    # convert to grey
    if (greyscale):
        grey_img = skimage.color.rgb2gray(img, channel_axis=channel_axis)
        grey_img[grey_img == 0] = np.mean(grey_img[grey_img > 0])
    
        # blur a little
        grey_img = skimage.filters.gaussian(grey_img, sigma=3)

        # if image is too big, scale it down
        scale_factor = 1
        max_input_img_axis_length = np.max(grey_img.shape)
        if (max_input_img_axis_length > max_image_length):
            scale_factor = max_image_length / max_input_img_axis_length
            grey_img = skimage.transform.rescale(
                grey_img, 
                scale=scale_factor, 
                anti_aliasing=False,
            )
            res *= scale_factor 
        
        # adjust for under-exposure/under-saturation
        grey_img = skimage.exposure.equalize_adapthist(grey_img)
        
        # otsu
        threshold_global_otsu = threshold_otsu(grey_img)
        global_otsu = (grey_img >= threshold_global_otsu)

        return res, orig_img, grey_img, global_otsu
    
    else: 
        # use grey_img only for thresholding
        grey_img = skimage.color.rgb2gray(img, channel_axis=channel_axis)

        # blur a little
        img = skimage.filters.gaussian(img, sigma=3, channel_axis=channel_axis)

        # if image is too big, scale it down
        scale_factor = 1
        max_input_img_axis_length = np.max(img.shape)
        if (max_input_img_axis_length > max_image_length):
            scale_factor = max_image_length / max_input_img_axis_length
            img = skimage.transform.rescale(
                img, 
                scale=scale_factor, 
                anti_aliasing=False,
                channel_axis=channel_axis
            )
            grey_img = skimage.transform.rescale(
                grey_img, 
                scale=scale_factor, 
                anti_aliasing=False,
            )
            res *= scale_factor 
        
        # adjust for under-exposure/under-saturation
        grey_img = skimage.exposure.equalize_adapthist(grey_img)

        # otsu
        threshold_global_otsu = threshold_otsu(grey_img)
        global_otsu = (grey_img >= threshold_global_otsu)

        return res, orig_img, img, global_otsu
