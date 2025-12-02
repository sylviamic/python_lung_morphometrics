import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import skimage
import sklearn.ensemble
import skops.io
import tifffile
import pandas as pd

import os
from joblib import Parallel, delayed
import functools

from ._utils import (
    _make_thresholded_images,
)

import importlib.resources

def make_kmeans_model_from_images(
    filenames: str | tuple,
    n_clusters: int = 4,
    initial_means: str | list = "k-means++",
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
    
    initial_means: list (default: None)
        Initial k-means cluster means, in LAB color space.

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
    
    # scatter the tasks (run the function across all images)
    func_args = []
    for f in filenames:
        func_args.append((f, {'do_sparse': do_sparse}))
    ret = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_get_pixel_samples)(*args, **kwargs) 
        for *args, kwargs in func_args
    )
    
    # gather the outputs
    pixel_samples = np.concatenate(ret, axis=0)
    
    # train the model
    if (n_clusters == 4):
        initial_means = (
            (1,-1), 
            (15,-8),
            (29,-19),
            (42,-29),
        )
    elif (n_clusters == 3):
        initial_means = ( 
            (15,-8),
            (29,-19),
            (42,-29),
        )
    else:
        initial_means = "k-means++"
    
    kmeans = sklearn.cluster.MiniBatchKMeans(
        n_clusters=n_clusters,
        init=initial_means,
        random_state=0,
        n_init='auto'
    ).fit(pixel_samples)
    
    return kmeans


def do_kmeans_cluster_image(
    img_path: str,
    pretrained_model: sklearn.cluster.KMeans = None,
    n_clusters: int = 4, 
    initial_means: str | list = "k-means++",
    do_sparse: bool = False,
    channel_axis: int = 0,
    return_stats: bool = True,
    save_fig: bool = True,
    save_tiff: bool = False,
    save_table: bool = True,
    save_dir: str = "./output/",
    dpi: int = 450,
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
    
    initial_means: list (default: None)
        Initial k-means cluster means, in LAB color space.

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
    
    save_fig: bool (default: True),
        If True, save an image with the cluster assignments
        overlaid on top of the original image.

    save_tiff: bool (default: False)
        If True, save binary masks of each region in a TIFF.
    
    save_table: bool (default: True)
        If True, save the results table as a TSV file in `save_dir`.

    save_dir: str (default: ./output/)
        The directory to save results to.

    dpi: int > 0 (default: 450)
        The DPI to save any figures at.
    
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
        np.delete(
            skimage.filters.gaussian(img, sigma=4, channel_axis=channel_axis),
            0, 
            channel_axis
        ), 
        (img.shape[xy_axes[0]] * img.shape[xy_axes[1]], 2)
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
            n_samples=int(img_array.shape[0]*0.1)
        )
        if (n_clusters == 4):
            initial_means = (
                (1,-1), 
                (15,-8),
                (29,-19),
                (42,-29),
            )
        elif (n_clusters == 3):
            initial_means = ( 
                (15,-8),
                (29,-19),
                (42,-29),
            )
        else:
            initial_means = "k-means++"
        
        kmeans = sklearn.cluster.MiniBatchKMeans(
            n_clusters=n_clusters,
            init=initial_means,
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
    lightness_img = img[:,:,0] 
    print(img.shape)
    print(lightness_img.shape)
    res = np.zeros(
        shape=(img.shape[xy_axes[0]], img.shape[xy_axes[1]], n_clusters)
    )
    cluster_mean_intensity = {}
    cluster_mean_a = {}
    cluster_mean_b = {}
    cluster_counts = {}
    for cluster in range(0, n_clusters):
        res[:,:,cluster] = np.squeeze(np.array(label_img == cluster, dtype=int))
        cluster_mean_intensity[cluster] = np.mean(lightness_img * res[:,:,cluster])
        cluster_mean_a[cluster] = np.mean(img[:,:,1] * res[:,:,cluster])
        cluster_mean_b[cluster] = np.mean(img[:,:,2] * res[:,:,cluster])
        cluster_counts[cluster] = np.sum(res[:,:,cluster] == cluster)
    
    # sort and name the clusters based on b
    #cluster_order = sorted(cluster_counts, key=lambda k: cluster_counts[k])[::-1]
    #if (n_clusters == 3):
    #    cluster_names  = ["Somewhat affected", "Unaffected",  "Affected"]
    #    cluster_colors = ["yellow", "blue",  "red"]
    #elif (n_clusters == 4):
    #    cluster_names  = ["Unaffected", "Somewhat affected", "Affected", "Background"]
    #    cluster_colors = ["blue", "yellow", "red", "lightgrey"]
    cluster_order = sorted(cluster_counts, key=lambda k: cluster_mean_b[k])[::-1]
    if (n_clusters == 3):
        cluster_names  = ["Unaffected", "Somewhat affected", "Affected"]
        cluster_colors = ["blue", "yellow",  "red"]
    elif (n_clusters == 4):
        cluster_names  = ["Background", "Unaffected", "Somewhat affected", "Affected"]
        cluster_colors = ["lightgrey", "blue", "yellow", "red"]
    
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
        colors=cluster_colors,
        bg_color="lightgrey"
    )

    # write the binary masks to a TIFF stack
    try:
        os.mkdir(save_dir)
    except Exception as e:
        print(e)
        pass

    if (save_tiff):
        save_filename = (
            save_dir + 
            os.path.basename(img_path).split(img_ext)[0] + 
            "_clustered.tiff"
        )
        save_me = np.swapaxes(res, 0, 2).astype('uint8') * np.iinfo('uint8').max
        save_me = np.swapaxes(save_me, 2,1)
        with tifffile.TiffWriter(save_filename, imagej=True) as tif:
            tif.write(save_me, photometric='minisblack', compression="JPEG")

    if (save_fig):
        # Plot the image result
        plt.subplot(211)
        plt.imshow(orig_img)
        plt.axis('off')
        plt.subplot(212)
        plt.imshow(res_composite)

        patches = []
        for i in range(0, len(cluster_names)):
            patches.append(matplotlib.patches.Patch(
                color=cluster_colors[i],
                label=cluster_names[i]
            ))
        plt.legend(handles=patches, loc="lower left", bbox_to_anchor=(0.25, 1.))
        plt.axis('off')

        plt.suptitle(os.path.basename(img_path))
        plt.tight_layout()
        
        save_filename = (
            save_dir + 
            os.path.basename(img_path).split(img_ext)[0] + 
            "_clustered.jpg"
        )
        plt.savefig(save_filename, dpi=dpi)
        plt.clf()

    if (save_table):
        save_filename = (
            save_dir + 
            os.path.basename(img_path).split(img_ext)[0] + 
            "_clustered_results.tsv"
        )
        df.to_csv(save_filename, sep="\t")

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
    channel_axis = np.argmin(img.shape)

    # flatten 2D image to 1D (and don't use L channel of image)
    xy_axes = [i for i in [0,1,2] if (i != channel_axis)]
    img_array = np.reshape(
        np.delete(
            skimage.filters.gaussian(img, sigma=4, channel_axis=channel_axis),
            0, 
            channel_axis
        ), 
        (img.shape[xy_axes[0]] * img.shape[xy_axes[1]], 2)
    )

    if (do_sparse):
        thresh_img_array = skimage.filters.rank.minimum(
            thresh_img.astype(int),
            skimage.morphology.disk(5)
        ).flatten().astype(np.bool_)
    else:
        thresh_img_array = np.ones_like(thresh_img, dtype=np.bool_).flatten()

    # take only non-background pixels
    filtered_img_array = img_array[thresh_img_array, :]

    # get random sample pixels from image
    pixel_samples = sklearn.utils.shuffle(
        filtered_img_array,
        random_state=0,
        n_samples=int(filtered_img_array.shape[0]*0.1)
    )
    
    return pixel_samples


def _get_multiscale_features(
    img: np.array,
    min_sigma: int = 1,
    max_sigma: int = 8,
    channel_axis: int = -1
):
    features_func = functools.partial(
        skimage.feature.multiscale_basic_features,
        intensity=True,
        edges=False,
        texture=True,
        sigma_min=min_sigma,
        sigma_max=max_sigma,
        channel_axis=channel_axis,
    )
    return features_func(img)

def train_feature_based_random_forest_model(
    img: str | np.ndarray,
    labels_path: str = None,
    save_dir: str = "output",
    save_model: bool = False,
    n_jobs=4
):
    if (type(img) is str):
        img = skimage.io.imread(img)

    greyscale_img = skimage.color.rgb2gray(img)
    threshold_img = greyscale_img > skimage.filters.threshold_otsu(greyscale_img)

    if (labels_path):
        training_labels = skimage.io.imread(labels_path)
    else:
        # assume it's the one known label img we have
        # hard-coded into this
        training_labels = np.zeros(img.shape[:2], dtype=np.uint8)

        training_labels[1010:1800, 2020:2500] = 2
        training_labels[260:470, 4300:4560]   = 3
        training_labels[1000:1140, 4650:4730] = 4

        training_labels = training_labels * threshold_img
        
        training_labels[70:1450, 70:480] = 1
        training_labels[2600:2700, 2000:4000] = 1

    features = _get_multiscale_features(img)

    clf_model = sklearn.ensemble.RandomForestClassifier(
        n_estimators=25, 
        n_jobs=n_jobs, 
        max_depth=8, 
        max_samples=0.05
    )
    clf_model = skimage.future.fit_segmenter(
        training_labels, 
        features, 
        clf_model
    )

    if (save_model):
        skops.io.dump(
            clf_model,
            os.path.join(save_dir, "he_feature_classifier.skops")
        )

    return clf_model

def apply_feature_based_random_forest_model(
    img_path: str,
    clf_model: sklearn.ensemble.RandomForestClassifier = None,
    save_dir: str = "output",
    save_fig: bool = True,
    save_tiffs: bool = False,
    save_table: bool = True,
    dpi: int = 300,
):
    img_ext = "." + img_path.split(".")[-1]

    # read and threshold the image
    res, orig_img, img, thresh_img = _make_thresholded_images(
        img_path, 
        greyscale=False,
        channel_axis=None
    )

    img[img == 0] = np.nan

    # get features (texture,color,blurs,etc.)
    features = _get_multiscale_features(img)

    # classify 
    if not (clf_model):
        plm_resources = importlib.resources.files("python_lung_morphometrics")
        model_path = plm_resources.joinpath(
            "models", 
            "default_he_features_damage_classification_model.skops"
        )
        clf_model = skops.io.load(model_path)

    classification_img = skimage.future.predict_segmenter(
        features, 
        clf_model
    )
    
    # set areas we know are background to background label
    classification_img[thresh_img] = 1

    # calculate statistics
    res_dict = {}
    row_label = os.path.basename(img_path)
    res_dict[row_label] = {}
    for i in range(1, np.max(classification_img) + 1):
        pct_key = "class_" + str(i) + "_pct"
        n_key   = "class_" + str(i) + "_n_pixels"
        res_dict[row_label][pct_key] = round(
            100 * np.sum(classification_img == i) / classification_img.size, 3
        )
        res_dict[row_label][n_key] = np.sum(classification_img == i)
    df = pd.DataFrame.from_dict(res_dict).T

    if (save_fig):
        fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(26,16))
        ax[0].imshow(
            skimage.segmentation.mark_boundaries(
                img, 
                classification_img, 
                mode="inner",
            )
        )
        ax[0].set_title('Image')
        ax[1].imshow(classification_img, cmap="jet")
        ax[1].set_title('Segmentation')
        fig.tight_layout()
        fig_savename = os.path.basename(img_path).replace(
            img_ext, "_classified.jpg"
        )
        fig.savefig(os.path.join(save_dir, fig_savename), dpi=dpi)

    if (save_tiffs):
        save_filename = os.path.join(
            save_dir + 
            os.path.basename(img_path).replace(img_ext, "_clustered.tif")
        )
        with tifffile.TiffWriter(save_filename, imagej=True) as tif:
            tif.write(
                classification_img.astype(np.uint8), 
                photometric='minisblack', 
                compression="lzw"
            )

    if (save_table):
        df_savename = os.path.basename(img_path).replace(
            img_ext, "_classification_results.tsv"
        )
        df.to_csv(
            os.path.join(save_dir, df_savename),
            sep = "\t"
        )

    return df