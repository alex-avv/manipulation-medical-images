from urllib.request import urlretrieve
import numpy as np
from time import time
from volume import Image3D, get_slices


def get_data():
    return np.load('3d_image.npy')


if __name__ == "__main__":
    npyData = get_data()
    mmVxDim = (2, 0.5, 0.5)
    mmSD = (2, 0.5, 0.5)
    vol = Image3D(npyData, mmVxDim)

    # EXPERIMENT 1
    scenarios = {'1-up-sampling': (1.8, 3, 1.66666),
                 '2-down-sampling': (0.4, 0.2, 0.33333),
                 '3-isotropic-voxels': (2, 0.5, 0.5)}

    for n, ratio in enumerate(scenarios.values(), 1):

        start_no_filter = time()
        resized_vol_no_filter = vol.volume_resize(ratio)
        end_no_filter = time()
        time_no_filter = end_no_filter - start_no_filter

        start_w_filter = time()
        resized_vol_w_filter = vol.volume_resize_antialias(ratio, mmSD)
        end_w_filter = time()
        time_w_filter = end_w_filter - start_w_filter

        get_slices(resized_vol_no_filter.data, resized_vol_no_filter.vx_dim,
                   filtering=False, m=1, n=n, timing=time_no_filter, save=True)
        get_slices(resized_vol_w_filter.data, resized_vol_no_filter.vx_dim,
                   filtering=True, m=1, n=n, timing=time_w_filter, save=True)

        if n == 1:
            vol_scenario1 = resized_vol_no_filter

    """
    Timing comments:
        The execution times when using both methods can be seen in the saved
        slices.
        After running the script multiple times, one can see how generally,
        resizing with volume_resize_antialias takes longer than with
        volume_resize. This makes sense as volume_resize_antialias involves the
        extra step of applying the Gaussian filter before interpolation.
        However, the time differences are not very significant and are in the
        order of 100s of ms.
    """

    # EXPERIMENT 2
    ratio = scenarios['1-up-sampling']
    ratio_inv = tuple(map(lambda a: 1 / a, ratio))
    # get_slices(vol.data, vol.vx_dim, filtering=False, m=2, more_info=(0, 0))

    # Resizing volume without filtering
    vol_no_filter = vol_scenario1.volume_resize(ratio_inv)
    diff_no_filter = ((vol.data - vol_no_filter.data) ** 2) ** 0.5
    diff_mean_no_filter = diff_no_filter.mean()
    diff_sd_no_filter = diff_no_filter.std()
    get_slices(vol_no_filter.data, vol_no_filter.vx_dim, filtering=False, m=2,
               more_info=(diff_mean_no_filter, diff_sd_no_filter), save=True)

    # Resizing volume with Gaussian filtering
    mmSD = (0.03, 0.1, 0.1)
    vol_filter = vol_scenario1.volume_resize_antialias(ratio_inv, mmSD)
    diff_filter = ((vol.data - vol_filter.data) ** 2) ** 0.5
    diff_mean_filter = diff_filter.mean()
    diff_sd_filter = diff_filter.std()
    get_slices(vol_filter.data, vol_filter.vx_dim, filtering=True, m=2,
               more_info=(diff_mean_filter, diff_sd_filter), save=True)

    """
    Mean and standard deviation of the voxel-level intensity differences:
        The differences between the original image and down-sampled images
        can be seen in the saved slices.

        The script was tested with the following up-sampling parameters and
        Gaussian standard deviations:
            Resize ratio: (1.8, 2.4, 17.3), Gaussian s.d: (0, 0, 0.01)
            Resize ratio: (1.3, 1.7, 9.4), Gaussian s.d: (0, 0, 0.01)
            Resize ratio: (16.6, 1.1, 2.9), Gaussian s.d: (0.01, 0, 0)
            Resize ratio: (1.3, 19.3, 1.5), Gaussian s.d: (0, 0.01, 0)
            Resize ratio: (1.3, 19.3, 1.5), Gaussian s.d: (0, 0.01, 0.01)
            Resize ratio: (2.1, 8.3, 1.2), Gaussian s.d: (0, 0.01, 0)
            Resize ratio: (1.8, 3, 1.66666), Gaussian s.d.: (0.03, 0.1, 0.1)
        The reason for up-sampling so much on one dimension was to investigate
        the effect of aliasing while running the script at moderate speed.

        In all the investigated cases, the intensity mean and standard
        deviation differences were larger using volume_resize_antialias than
        using volume_resize. This is intuitive, as Gaussian filtering involves
        information loss; except in aliasing situations. In such scenarios,
        applying a Gaussian filter before down-sampling should remove the
        distortion and give lower differences compared to the original image.
    """
