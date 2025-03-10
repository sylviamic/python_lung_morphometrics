import numpy as np
import matplotlib.pyplot as plt
import sklearn
import skimage
import imageio
import tifffile
import pandas as pd

import glob
import os
from joblib import Parallel, delayed

from ._utils import (
    _make_thresholded_images,
)


def make_kmeans_model_from_images(
    filenames: str | tuple,
    n_clusters: int = 4,
    do_sparse: bool = False, 
    channel_axis: int = None,
    n_jobs: int = 4,
):

    """
    Given the path to a set of H&E images, this 
    function will concatenate the images, 
    convert the joint image to LAB colorspace,
    give it a light blur, then use KMeans 
    clustering (k=4) to identify regions of
    interest and return the model.
        
    Parameters
    ----------
    filenames: str
        The path(s) to the image files in
        .tiff/.tif format
            
    n_clusters: int > 0  (default: 4)
        The number of clusters to split the 
        image into
    
    do_sparse: bool (default: False)
        If True, use only thresholded pixels
        as potential training pixels. If False,
        consider all pixels (simpler).

    channel_axis: int (default: None)
        The axis corresponding to color channels.

    n_jobs: int > 0 (default: 4)
        Split the images across multiple 
        processes in parallel; set to 1 
        to disable parallel processing.
            
    Returns
    -------
    model: sklearn.cluster.KMeans object (optional)
        The trained KMeans clustering model
        
    """
    
    # locate the images for training
    #filenames =  glob.glob(img_path + "/*.tif")
    #filenames += glob.glob(img_path + "/*.tiff")
        
    # scatter the tasks (run the function across all images)
    func_args = []
    for f in filenames:
        func_args.append((f, {'do_sparse': False}))
    ret = Parallel(n_jobs=n_jobs, verbose=10)(
        #delayed(_get_pixel_samples)(f) for f in filenames
        delayed(_get_pixel_samples)(*args, **kwargs) 
        for *args, kwargs in func_args
    )
    
    # gather the outputs
    pixel_samples = np.concatenate(ret, axis=0)
    
    # train the model
    kmeans = sklearn.cluster.MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init='auto'
    ).fit(pixel_samples)
    
    return kmeans


def cluster_image(
    img_path: str,
    save_path: str = "./clustered/",
    pretrained_model: sklearn.cluster.KMeans = None,
    n_clusters: int = 4, 
    do_sparse: bool = False,
    channel_axis: int = 0,
    return_stats: bool = True,
    save_tiffs: bool = False
):
    """
    Given the filename of an H&E image, this 
    function will convert it to LAB colorspace,
    give it a light blur, then use KMeans 
    clustering (k=4) to identify regions of
    interest and generate a TIFF stack of binary
    masks, 1 channel for each region (cluster).
    
    Parameters
    ----------
    img_path: str
        The path to the image file
    
    pretrained_model: sklearn.cluster.MiniBatchKMeans
        A pretrained, instantiated KMeans model
        to be used for classification. Incompatible
        with n_clusters. Use this is you already trained
        a reliable model on a sample image set
        and wish to reuse that classifier here.
        
    n_clusters: int > 0  (default: 4)
        The number of clusters to split the 
        image into
    
    do_sparse: bool (default: False)
        If True, perform processing only on pixels
        found in the foreground of the image after
        basic thresholding. If False, process all
        pixels (simpler). 
    
    channel_axis: int (default: 0)
        The axis corresponding to color channels.

    return_stats: bool (default: True)
        Returns a len(1) dataframe of statistics for
        this image.
        
    save_tiffs: bool (default: False)
        Will save binary masks of each region in TIFF
        format if True
    
    Returns
    -------
    ret: ndarry of np.uint8
        The resultant binary masks, with shape
        [n_pixels_width, n_pixels_height, n_clusters]
    
    df: pd.DataFrame
        Dataframe containing # of pixels in 
        each identified cluster
        
    """
    
    # load the image from disk
    img_ext = "." + img_path.split(".")[-1]
    
    res, orig_img, img, thresh_img = _make_thresholded_images(
        img_path,
        greyscale=False
    )
    
    if not (channel_axis):
        if (len(orig_img.shape) < 3):
            raise ValueError(f"Image too small to contain channel axis! {img.shape}")
        else:
            channel_axis = np.argmin(orig_img.shape)

    # convert to LAB
    img = skimage.color.rgb2lab(img, channel_axis=channel_axis)
    lab_channel_axis = np.argmin(img.shape)

    # flatten from 2D to 1D for classification
    xy_axes = [i for i in [0,1,2] if (i != lab_channel_axis)]
    img_array = np.reshape(
        img, 
        (img.shape[xy_axes[0]] * img.shape[xy_axes[1]], 3)
    )
    if (do_sparse):
        thresh_img = ~thresh_img
    else:
        thresh_img = np.ones_like(thresh_img, dtype=np.bool_)
    thresh_img_array = thresh_img.flatten()

    if (pretrained_model):
        kmeans = pretrained_model
    else:
        # train the model on a subset of the data, with many random samples
        img_array_sample = sklearn.utils.shuffle(
            img_array[thresh_img_array],
            random_state=0, 
            n_samples=int(img_array.shape[xy_axes[0]]*0.2)
        )
        kmeans = sklearn.cluster.MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init='auto'
        ).fit(img_array_sample)

    labels = kmeans.predict(img_array[thresh_img_array])

    # put predicted labels back into image 
    label_img = np.full_like(img_array[:,0], np.nan, int)
    np.place(label_img, thresh_img_array, labels)
    label_img = np.reshape(
        label_img, 
        (img.shape[xy_axes[0]], img.shape[xy_axes[1]], 1)
    )

    # arrange result image mask and calculate statistics 
    lightness_img = img[:,:,0] #np.take(img, lab_channel_axis, 0) # img[:,:,0] # L* channel
    print(img.shape)
    print(lightness_img.shape)
    res = np.zeros(
        shape=(img.shape[xy_axes[0]], img.shape[xy_axes[1]], n_clusters)
    )
    cluster_mean_intensity = {}
    cluster_counts = {}
    for cluster in range(0, n_clusters):
        res[:,:,cluster] = np.squeeze(np.array(label_img == cluster, dtype=int))
        cluster_mean_intensity[cluster] = np.mean(lightness_img * res[:,:,cluster])
        cluster_counts[cluster] = np.sum(res[:,:,cluster] == cluster)
        #cluster_counts[cluster] = np.unique(res[:,:,cluster], return_counts=True)[1]
    
    # sort and name the clusters based on intensity
    cluster_order = sorted(cluster_counts, key=lambda k: cluster_counts[k])[::-1]
    if (n_clusters == 3):
        cluster_names = ["Unaffected", "Somewhat affected", "Affected"]
    elif (n_clusters == 4):
        cluster_names = ["Unaffected", "Somewhat affected", "Affected", "Background"]
    
    cluster_name_dict = {x: cluster_names[i] for i,x in enumerate(cluster_order)}
    for i,x in enumerate(cluster_order):
        labels[labels==x] = i + np.max(cluster_order) + 1
    labels -= np.max(cluster_order)
    
    # put predicted labels back into image 
    label_img = np.full_like(img_array[:,0], 0, dtype=int)
    np.place(label_img, thresh_img_array, labels)
    label_img = np.reshape(
        label_img, 
        (img.shape[xy_axes[0]], img.shape[xy_axes[1]])
    )

    # calculate the scores
    df = pd.DataFrame(index=[os.path.basename(img_path)])
    for key, val in cluster_name_dict.items():
        if ("Background" not in val):
            df[val + " area (px)"] = np.sum(res == key)
    df["Lung area (px)"] = df.sum(axis=1)
    for val in cluster_name_dict.values():
        if ("Background" not in val):
            df[val + " area (% of lung)"] = (
                df[val + " area (px)"] * 100 / df["Lung area (px)"]
            )

    res_composite = skimage.color.label2rgb(
        label_img,
        image=None,#skimage.color.rgb2gray(img), 
        bg_label=0,
        colors=["blue", "yellow", "red", "black"],
        bg_color="white"
    )

    # write the binary masks to a TIFF stack
    try:
        os.mkdir(save_path)
    except Exception as e:
        print(e)
        pass

    if (save_tiffs):
        save_filename = (
            save_path + 
            os.path.basename(img_path).split(img_ext)[0] + 
            "_clustered.tiff"
        )
        save_me = np.swapaxes(res, 0, 2).astype('uint8') * np.iinfo('uint8').max
        save_me = np.swapaxes(save_me, 2,1)
        with tifffile.TiffWriter(save_filename, imagej=True) as tif:
            tif.write(save_me, photometric='minisblack', compression="JPEG")

    # Plot the image result
    plt.subplot(211)
    plt.imshow(orig_img)
    plt.axis('off')
    plt.subplot(212)
    plt.imshow(res_composite)
    plt.axis('off')

    plt.suptitle(os.path.basename(img_path))
    plt.tight_layout()
    
    save_filename = (
        save_path + 
        os.path.basename(img_path).split(img_ext)[0] + 
        "_clustered.png"
    )
    plt.savefig(save_filename, dpi=250)
    plt.clf()

    # return the image and other objects as requested
    ret = [res]
    if (return_stats):
        ret.append(df)
    del img, orig_img, lightness_img, res_composite, labels, label_img, img_array
        
    if (len(ret) > 1):
        return ret
    else:
        return res


def _get_pixel_samples(
    img_path: str,
    do_sparse: bool = False,
    channel_axis: int = None,
):
    """
    Given a path to an image, load it, contrast-adjust it,
    threshold it, convert it to LAB colorspace, and then identify
    a fraction of pixels to use as sample data for training a
    pixel classifier.

    Parameters
    ----------
    
    img_path: str
        The path to the image to sample
    
    do_sparse: bool (default: False)
        If True, only take pixels that have been thresholded 
        from the background for training. If False, consider
        all pixels in image when sampling (simpler).

    channel_axis: int (default: 0)
        The axis corresponding to color channels.

    Returns
    -------

    pixel_samples: np.ndarray

    """
    res, orig_img, img, thresh_img = _make_thresholded_images(
        img_path, 
        greyscale=False,
        channel_axis=channel_axis
    )
    
    if not (channel_axis):
        if (len(orig_img.shape) < 3):
            raise ValueError(f"Image too small to contain channel axis! {img.shape}")
        else:
            channel_axis = np.argmin(orig_img.shape)

    # convert to LAB
    img = skimage.color.rgb2lab(img, channel_axis=channel_axis)
    
    # flatten 2D image to 1D
    xy_axes = [i for i in [0,1,2] if (i != channel_axis)]
    img_array = np.reshape(
        img, 
        (img.shape[xy_axes[0]] * img.shape[xy_axes[1]], 3)
    )

    if (do_sparse):
        thresh_img_array = thresh_img.flatten()
    else:
        thresh_img_array = np.ones_like(thresh_img, dtype=np.bool_).flatten()

    # take only non-background pixels
    filtered_img_array = img_array[thresh_img_array, :]

    # get random sample pixels from image
    pixel_samples = sklearn.utils.shuffle(
        filtered_img_array,
        random_state=0,
        n_samples=int(filtered_img_array.shape[0] * 0.2)
    )
    
    return pixel_samples