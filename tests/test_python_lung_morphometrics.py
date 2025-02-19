#!/usr/bin/env python

import pytest
import os

import pandas as pd
import numpy as np

import python_lung_morphometrics as plm

@pytest.fixture
def get_data_dir():
    return os.path.join("tests", "data")

@pytest.fixture
def get_mli_img_filepath():
    return os.path.join("tests", "data", "2086_20X_DISTAL6_ch00.TIFF")

@pytest.fixture
def get_coloc_img_filepath():
    return os.path.join(
        "tests", 
        "data", 
        "20250130_2327pt3_DAPI405_EdU488_LAMP3647_small_crop.tif"
    )
@pytest.fixture
def get_coloc_mask_img_filepath():
    return os.path.join(
        "tests", 
        "data", 
        "20250130_2327pt3_DAPI405_EdU488_LAMP3647_small_crop_cp_masks.tif"
    )


@pytest.fixture
def get_injury_img_filepath():
    return (
        os.path.join("tests", "data", "p601g_1b.tif"),
        os.path.join("tests", "data", "2549_1a.tif"),
    )
'''
@pytest.fixture
def train_kmeans_model(
    get_data_dir,
    n_clusters = 3,
    n_jobs = 1
):
    return plm.make_kmeans_model_from_images(
        get_data_dir, 
        n_clusters=n_clusters, 
        n_jobs=n_jobs
    )
'''

def test_do_mli(
    get_mli_img_filepath
): 
    res = plm.do_mli(get_mli_img_filepath)

    assert ((res > 5) and (res < 300))

'''
def test_make_kmeans_model_from_images(
    train_kmeans_model,
):
    assert len(train_kmeans_model.labels_) > 0
    assert max(train_kmeans_model.labels_) < 4
    assert min(train_kmeans_model.labels_) > -1


def test_cluster_image(
    get_injury_img_filepath,
    train_kmeans_model,
):
    #m = train_kmeans_model
    clustered_healthy_img, df_healthy = plm.cluster_image(
        get_injury_img_filepath[0],
        pretrained_model = train_kmeans_model
    )
    clustered_injured_img, df_injured = plm.cluster_image(
        get_injury_img_filepath[1],
        pretrained_model = None,
        n_clusters = 4
    )

    df = pd.concat([df_healthy, df_injured])

    print(df.head())
    assert len(df) == 2
    assert len(df.columns) > 1
'''

def test_do_colocalization_analysis(
    get_coloc_img_filepath,
    get_coloc_mask_img_filepath,
):
    df = plm.do_colocalization_analysis(
        get_coloc_img_filepath,
        nuc_seg_img_filename=get_coloc_mask_img_filepath,
        use_cellpose=False,
        use_gpu=False
    )

    assert len(df) > 1

    # roughly 80 EdU+ nuclei
    assert 50 < np.sum(df["intensity_mean_1"] > 0.4) < 100

    # roughly 250 LAMP3+ cells
    assert 150 < np.sum(df["intensity_mean_2"] > 0.2) < 300