import skimage
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

has_cellpose_library = False
try:
    from cellpose import models
    has_cellpose_library = True
except ImportError as e:
    pass

import imageio
import tifffile
import os

def _threshold_signal(signal_img):
    smooth = skimage.filters.gaussian(signal_img, sigma=1)
    signal_thresh_img = smooth > skimage.filters.threshold_otsu(smooth)
    return signal_thresh_img

def _threshold_nuclei(
    nuc_img, 
    threshold_method="otsu"
):
    smooth = skimage.filters.gaussian(nuc_img, sigma=0.1)
    
    if (threshold_method == "otsu"):
        thresh = smooth > skimage.filters.threshold_otsu(smooth)
    elif (threshold_method == "multiotsu"):
        smooth = skimage.exposure.equalize_adapthist(
            smooth, 
            clip_limit=0.05,
            kernel_size = (int(smooth.shape[0]/8), int(smooth.shape[1]/8)),
        )
        thresholds = skimage.filters.threshold_multiotsu(smooth, classes=3)
        thresh = smooth > thresholds[-1]
    elif (threshold_method == "sauvola"):
        thresh = smooth > skimage.filters.threshold_sauvola(smooth, window_size=15)

    fill = scipy.ndimage.binary_fill_holes(thresh)
    nuc_thresh_img = skimage.segmentation.clear_border(fill)
    return nuc_thresh_img

def _segment_nuclei(nuc_thresh_img):
    distance = scipy.ndimage.distance_transform_edt(nuc_thresh_img)
    coords = skimage.feature.peak_local_max(
        distance,
        footprint=np.ones((3, 3)),
        labels=nuc_thresh_img
    )
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = scipy.ndimage.label(mask)
    nuc_seg_img = skimage.segmentation.watershed(
        -1 * distance, 
        markers, 
        mask=nuc_thresh_img
    )
    return nuc_seg_img

def _expand_nucleus_seg(
    nuc_seg_img, 
    expansion_factor=1.2
):
    properties_to_measure = [
        "label",
        "equivalent_diameter_area"
    ]
    regionprops = skimage.measure.regionprops_table(
        nuc_seg_img, 
        separator="_",
        properties=properties_to_measure
    )
    regionprops_df = pd.DataFrame(regionprops)
    print(regionprops_df.columns)
    mean_effective_diameter = np.mean(regionprops_df["equivalent_diameter_area"])
    distance_to_expand = np.max(
        [np.round((1.0 - expansion_factor) * 0.5 * mean_effective_diameter), 1]
    )

    nuc_seg_expanded = skimage.segmentation.expand_labels(
        nuc_seg_img,
        distance_to_expand
    )
    return nuc_seg_expanded


def _measure_signal_seg_overlap_with_nucleus_seg(
    nuc_seg_img,
    signal_thresh_img
):
    properties_to_measure = [
        "label",
        "area",
        "centroid",
        "intensity_max",
        "intensity_min",
        "intensity_mean",
        "equivalent_diameter_area"
    ]
    regionprops = skimage.measure.regionprops_table(
        nuc_seg_img, 
        signal_thresh_img,
        separator="_",
        properties=properties_to_measure
    )
    regionprops_df = pd.DataFrame(regionprops)
    return regionprops_df

def _set_slice_of_ndarray(x, axis, idx, vals):
    # x - original image (ndarray)
    # axis - position of axis to insert into
    # idx - index of axis to insert into
    # vals - values to insert into x[...,axis,...][idx]

    slices = [slice(None)] * x.ndim
    slices[axis] = idx
    x[tuple(slices)] = vals
    return None


def do_colocalization_analysis(
    img_filename,
    nuc_seg_img_filename=None,
    use_cellpose=False, 
    use_gpu=False,
    channel_axis=0,
    nucleus_channel_idx=0,
    save_table=True,
    save_intermediate_images=True,
    dpi=450,
    save_dir="output"
):
    """
    Given the path to a tiff file, 
    segment out the nuclei and measure fraction of 
    thresholded signal overlap across each nuclear ROI. 

    Parameters
    ----------

    img_filename
        The full path to the tiff image

    nuc_seg_img_filename: str (default: None)
        If the nuclei have already been segmented, the full
        path to this segmentated tiff image (i.e. cellpose output)

    use_cellpose: bool (default: False)
        Segment nuclei using cellpose. Requires `cellpose` python
        library to be installed. Will download nuclei model to cellpose
        default storage directory if not found there. See cellpose docs.

    use_gpu: bool (default: False)
        If using cellpose, determines whether or not to attempt to
        use the GPU for nuclei segmentation (much faster than CPU).
    
    channel_axis: int >= 0 (default: 0)
        The axis of the np.ndarray representation of the img_filename
        that corresponds to signal (color) channels.

    nucleus_channel_idx: int >=0 (default: 0)
        Which position along channel_axis corresponds to signal for
        nuclear stain (DAPI/Hoescht/DRAQ5/etc.)

    save_table: bool (default: True)
        Save the ROI regionprops results table to the working directory 
        as a tab-separated file.
    
    save_intermediate_images: bool (default: True)
        Save an SVG file containing plots of the original signal intensities, 
        the thresholded channel values, and the nuclear segmentation overlaid
        atop the original signal. Very useful for QC.

    dpi: int (default: 450)
        DPI of output images

    save_dir: str (default: "output")
        Path to save the output tables/images to. Defaults to ./output, a 
        subdirectory of the current working directory which will be created
        if it does not exist.


    Returns
    -------

    df: pd.DataFrame
        The results table.

    """

    # load the image
    img = skimage.img_as_ubyte(
        imageio.v3.imread(img_filename)
    )  
    img_ext = "." + img_filename.split(".")[-1]

    n_axes = len(img.shape)

    nuc_img = np.take(img, nucleus_channel_idx, axis=channel_axis)
    if (nuc_seg_img_filename):
        # load pre-generated nuclei segmentation mask
        nuc_seg_img = skimage.img_as_uint(
            imageio.v3.imread(nuc_seg_img_filename)
        )
        nuc_thresh_img = nuc_seg_img > 0
    elif (use_cellpose and has_cellpose_library):
        # generate nuclei segmentation mask with cellpose
        model = models.Cellpose(gpu=use_gpu, model_type='nuclei')
        nuc_seg_img, _,_,_ = model.eval(
            nuc_img, 
            diameter=None, 
            channels=[0,0],
            flow_threshold=0,
            cellprob_threshold=-3,
            do_3D=False
        )
        nuc_thresh_img = nuc_seg_img > 0
    else:
        # try and generate a nuclei segmentation mask 
        # using thresholding + watershed (poor performance)
        nuc_thresh_img = _threshold_nuclei(nuc_img)
        nuc_seg_img = _segment_nuclei(nuc_thresh_img) 
    
    # expand the nuclei ROIs just a bit
    nuc_seg_expanded_img = _expand_nucleus_seg(nuc_seg_img)

    signal_thresh_img = np.zeros_like(
        np.take(img, range(0,img.shape[channel_axis]), axis=channel_axis)
    )

    # TODO: generalize to N-dimensional images
    for i in range(0, img.shape[channel_axis]):
        if (i == nucleus_channel_idx):
            _set_slice_of_ndarray(
                signal_thresh_img, 
                axis=channel_axis, 
                idx=i, 
                vals=nuc_thresh_img
            )
        else:
            _set_slice_of_ndarray(
                signal_thresh_img, 
                axis=channel_axis, 
                idx=i, 
                vals=_threshold_signal(np.take(img, i, axis=channel_axis))
            )

    df = _measure_signal_seg_overlap_with_nucleus_seg(
        nuc_seg_expanded_img,
        np.moveaxis(signal_thresh_img, channel_axis, -1)
    )

    if (save_table):
        if not (os.path.exists(save_dir)):
            os.makedirs(save_dir)
        df.to_csv(
            os.path.join(
                save_dir,
                os.path.basename(img_filename) + "_coloc_results.tsv"
            ),
            sep="\t",
            index=False
        )

    if (save_intermediate_images):

        figure_scale = 4
        n_rows = 3
        n_cols = img.shape[channel_axis]
        cmap = plt.cm.Grays_r
        figsize_x, figsize_y = 3*n_rows, n_cols*figure_scale
        fig, axs = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_x, figsize_y),
            gridspec_kw = {"wspace": 0.05, "hspace": 0.05}
        )
        for i in range(0, n_rows):
            if (i == 0):
                x = img
                title = "intensity-channel-"
            elif (i == 1):
                x = signal_thresh_img
                title = "threshold-channel-"
            elif (i == 2):
                x = img
                y = nuc_seg_expanded_img
                title = "segmented-channel-"

            for j in range(0, n_cols):
                if (i == 2):
                 axs[i,j].imshow(
                    skimage.color.label2rgb(
                        y, 
                        image=np.take(x, j, axis=channel_axis), 
                        bg_label=0,
                        alpha=0.15
                    )
                )
                else:
                    axs[i,j].imshow(np.take(x, j, axis=channel_axis), cmap=cmap)
                axs[i,j].set_title(title + str(j))

        for a in axs.ravel():
            a.set_axis_off()
        
        plt.suptitle(img_filename)
        
        # save the image
        if not (os.path.exists(save_dir)):
            os.makedirs(save_dir)
        fig.savefig(
            os.path.join(
                save_dir, 
                os.path.basename(img_filename).replace(img_ext, "_coloc_output.svg")
            ), 
            dpi=dpi
        )
        plt.clf()
    return df