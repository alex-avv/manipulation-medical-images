from scipy import ndimage
import numpy as np
from time import time, strftime, gmtime
from multiprocessing import Pool
import matplotlib.pyplot as plt
from utils import show_slice, slice_at, gaussian_3d, bilateral_filter


def filtering(data, bilateral_filter, kernel_size, kwd_args):
    return ndimage.generic_filter(data, bilateral_filter, kernel_size,
                                  extra_keywords=kwd_args)


if __name__ == "__main__":
    ## Loading 3D image, previously read with SimpleITK and saved as a .npy
    ## file
    # import SimpleITK as sitk
    # vol_id = sitk.ReadImage('USProstate.gipl', imageIO='GiplImageIO')
    # vol = sitk.GetArrayFromImage(vol_id)
    # np.save('volume', vol)
    vol = np.load('volume.npy')

    ## Reading metadata from the original image. The voxel dimensions are
    ## 0.2 mm, 0.2 mm and 2 mm for the left, posterior and superior axes,
    ## respectively
    # vox_dims = vol_id.GetSpacing()

    # Flipping the volume so the anterior coordinates are increasing with the
    # array: in the original image volume, these decrease as you go along the
    # array 1-dim axis.
    # vol = np.flip(vol, axis=1)

    # ~~~ VOLUME RE-SLICING ~~~
    params = dict(axis='z', trans=0, x_rot=-4, y_rot=-0.5, z_rot=0.5)
    params2 = dict(axis='x', trans=-70, x_rot=0, y_rot=45, z_rot=0)
    params3 = dict(axis='y', trans=-45, x_rot=20, y_rot=0, z_rot=-20)

    # Getting slices from the volume with the specified parameters
    img = slice_at(vol, **params)
    img2 = slice_at(vol, **params2)
    img3 = slice_at(vol, **params3)

    # Showing the slices in 3D and 2D
    print('Getting slices...', end='')
    show_slice(img, **params, filtered='no', visualisation='3d',
               volume=vol, resol_3d=1, save=True)
    show_slice(img, **params, filtered='no', visualisation='2d', save=True)
    show_slice(img2, **params2, filtered='no', visualisation='3d',
               volume=vol, resol_3d=1, save=True)
    show_slice(img2, **params2, filtered='no', visualisation='2d', save=True)
    show_slice(img3, **params3, filtered='no', visualisation='3d',
               volume=vol, resol_3d=1, save=True)
    show_slice(img3, **params3, filtered='no', visualisation='2d', save=True)
    print('Done!')

    # ~~~ 3D BILATERAL FILTERING ~~~
    # Specifying geometric and photometric spreads for bilateral filter
    sigma_d = 3
    sigma_r = 2.5

    kernel_size = (3, 21, 21)
    gaussian_kernel_3d = gaussian_3d(sigma_d, kernel_size)
    gaussian_kernel_3d_flatten = np.ravel(gaussian_kernel_3d)
    kwd_args = dict(sigma_r=sigma_r, gaussian=gaussian_kernel_3d_flatten)

    ## Filtering without splitting the array
    ## Duration was 18:51 at time of run
    # print('Filtering...', end=' ')
    # start = time()
    # vol_filtered = filtering(vol, bilateral_filter, kernel_size, kwd_args)
    # end = time()
    # dur = strftime('%M:%S', gmtime(end - start))
    # print(f'Done!\nDuration = {dur}')

    # Filtering splitting the array
    # Duration was 8:01 at time of run
    pool = Pool(3)
    vol_bot = vol[0:16]
    vol_mid = vol[14:31]
    vol_top = vol[29:46]

    print('Filtering...', end=' ')
    start = time()
    results = pool.starmap(filtering, [(vol_bot, bilateral_filter, kernel_size,
                                        kwd_args),
                                       (vol_mid, bilateral_filter, kernel_size,
                                        kwd_args),
                                       (vol_top, bilateral_filter, kernel_size,
                                        kwd_args)])
    vol_bot_fld, vol_mid_fld, vol_top_fld = results[0], results[1], results[2]

    # Keeping only centre sections of the array unaffected by edges
    vol_bot_fld = vol_bot_fld[:15]
    vol_mid_fld = vol_mid_fld[1:16]
    vol_top_fld = vol_top_fld[1:]

    vol_filtered_split = np.vstack((vol_bot_fld, vol_mid_fld, vol_top_fld))
    end = time()
    dur_split = strftime('%M:%S', gmtime(end - start))
    print(f'Done!\nDuration = {dur_split}')

    # np.testing.assert_equal(vol_filtered, vol_filtered_split)
    vol_filtered = vol_filtered_split
    img_3d_filtered = slice_at(vol_filtered, **params)
    show_slice(img_3d_filtered, **params, filtered='3d', visualisation='2d',
               save=True)

    # ~~~ 2D BILATERAL FILTERING ~~~
    gaussian_kernel_2d = slice_at(gaussian_kernel_3d, **params, kernel=True)
    # show_slice(gaussian_kernel_2d, **params, filtered='no',
    #            visualisation='2d')
    gaussian_kernel_2d_flatten = np.ravel(gaussian_kernel_2d)
    kwd_args = dict(sigma_r=sigma_r, gaussian=gaussian_kernel_2d_flatten)

    print('Filtering...', end=' ')
    img_2d_filtered = filtering(img, bilateral_filter,
                                gaussian_kernel_2d.shape, kwd_args)
    print('Done!')
    show_slice(img_2d_filtered, **params, filtered='2d', visualisation='2d',
               save=True)

    # ~~~ QUANTITATIVE COMPARISON ~~~
    # Calculating voxel-level intensity differences with respect to un-filtered
    # slice
    diff_2d_fld = ((img - img_2d_filtered) ** 2) ** 0.5
    diff_mean_2d_fld = diff_2d_fld.mean()
    diff_sd_2d_fld = diff_2d_fld.std()

    diff_3d_fld = ((img - img_3d_filtered) ** 2) ** 0.5
    diff_mean_3d_fld = diff_3d_fld.mean()
    diff_sd_3d_fld = diff_3d_fld.std()

    quant_comparison = ('\u0332'.join('QUANTITATIVE_COMPARISON') + '\n'
                        f'Mean of un-filtered slice: {img.mean():.2f}\n'
                        f'S.d. of un-filtered slice: {img.std():.2f}\n\n'
                        'Mean of 3d-filtered slice: '
                        f'{img_3d_filtered.mean():.2f}\n'
                        'S.d. of 3d-filtered slice: '
                        f'{img_3d_filtered.std():.2f}\n\n'
                        'Mean of 2d-filtered slice: '
                        f'{img_2d_filtered.mean():.2f}\n'
                        'S.d. of 2d-filtered slice: '
                        f'{img_2d_filtered.std():.2f}\n\n'
                        f'Diff. mean for 3d-filtered: {diff_mean_3d_fld:.4g}\n'
                        f'Diff. s.d. for 3d-filtered: {diff_sd_3d_fld:.4g}\n\n'
                        f'Diff. mean for 2d-filtered: {diff_mean_2d_fld:.4g}\n'
                        f'Diff. s.d. for 2d-filtered: {diff_sd_2d_fld:.4g}\n\n'
                        'Diff. mean of the 2d-filtered slice is '
                        f'''{(diff_mean_2d_fld - diff_mean_3d_fld)
                             / diff_mean_3d_fld * 100:.4g}'''
                        ' % larger than that of 3d-filtered\n'
                        'Diff s.d. of the 2d-filtered slice is '
                        f'''{(diff_sd_2d_fld - diff_sd_3d_fld)
                             / diff_sd_3d_fld * 100:.4g}'''
                        ' % larger than that of 3d-filtered\n\n')

    def average_1d_spectrum(image):
        # Getting the real part of the centre-shifted Fourier Transform
        image_fft = np.fft.fft2(image)
        image_fft = np.fft.fftshift(image_fft)
        image_fft = np.abs(image_fft)

        def radial_profile(ctrd_2d_fft, center):
            y, x = np.indices((ctrd_2d_fft.shape))
            r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            r = r.astype('int')

            t_bin = np.bincount(r.ravel(), ctrd_2d_fft.ravel())
            n_r = np.bincount(r.ravel())
            profile = t_bin / n_r
            return profile

        return radial_profile(image_fft, (image_fft.shape[1] // 2,
                                          image_fft.shape[0] // 2))

    img_spectrum = average_1d_spectrum(img)
    img_2d_fld_spectrum = average_1d_spectrum(img_2d_filtered)
    img_3d_fld_spectrum = average_1d_spectrum(img_3d_filtered)

    plt.figure()
    plt.plot(np.log(img_spectrum))
    plt.plot(np.log(img_3d_fld_spectrum))
    plt.plot(np.log(img_2d_fld_spectrum))
    plt.ylabel('Log(Amplitude)')
    plt.xlabel('Frequency')
    plt.ylim([4, 16])
    plt.legend(('Not filtered', '3D filtered', '2D filtered'))
    plt.savefig('average-spectrums.png')

    img_3d_fld_l_f = np.trapz(img_3d_fld_spectrum[:90])
    img_3d_fld_m_f = np.trapz(img_3d_fld_spectrum[90:180])
    img_3d_fld_h_f = np.trapz(img_3d_fld_spectrum[180:])
    img_2d_fld_l_f = np.trapz(img_2d_fld_spectrum[:90])
    img_2d_fld_m_f = np.trapz(img_2d_fld_spectrum[90:180])
    img_2d_fld_h_f = np.trapz(img_2d_fld_spectrum[180:])

    quant_comparison += ('2d-filtered slice has '
                         f'''{(img_3d_fld_l_f - img_2d_fld_l_f) /
                              img_3d_fld_l_f * 100:.4g}'''
                         ' % less low-frequency components than 3d-filtered '
                         'one\n'
                         '2d-filtered slice has '
                         f'''{(img_3d_fld_m_f - img_2d_fld_m_f) /
                              img_3d_fld_m_f * 100:.4g}'''
                         ' % less medium-frequency components than 3d-filtered'
                         'one\n'
                         '2d-filtered slice has '
                         f'''{(img_2d_fld_h_f - img_3d_fld_h_f) /
                              img_3d_fld_h_f * 100:.4g}'''
                         ' % more high-frequency components than 3d-filtered '
                         'one\n')

    print(quant_comparison)
