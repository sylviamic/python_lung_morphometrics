#!/usr/bin/env python

"""Tests for `python_lung_morphometrics` package."""

import pytest
import os

from python_lung_morphometrics.python_lung_morphometrics import do_mli as _do_mli

img_filename = os.path.join(
    "tests", 
    "data", 
    "2086_20X_DISTAL6_ch00.tif"
)

@pytest.fixture
def do_mli():
    return _do_mli(
        img_filename
    )


def test_do_mli(do_mli):
    # Act
    res = do_mli

    # Assert
    assert ((res > 5) and (res < 300))