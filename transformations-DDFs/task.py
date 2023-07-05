from urllib.request import urlretrieve
import numpy as np
from transformations import RigidTransform, get_slices
from typing import Tuple, Dict


def get_data():
    return np.load('3d_image.npy')


def randomise(range_dict: Dict[str, Tuple[float, float]]):
    parameters = []
    for lower, higher in range_dict.values():
        parameters.append(np.random.uniform(lower, higher))
    return parameters


if __name__ == "__main__":
    print('Getting data...')
    npyData = get_data()

    # Padding data to avoid loosing information if volume moves out the main
    # frame. Note that padding data makes the code work slower than normal
    padz, pady, padx = 8, 32, 32
    npyData = np.pad(npyData, [(padz, padz), (pady, pady), (padx, padx)])

    def main_data(volume):
        return volume[padz:padz + 31, pady:pady + 127, padx:padx + 127]

    # EXPERIMENT 1
    print('\n' + '\u0332'.join('EXPERIMENT_1'))
    '''
    image_train00.npy is 32 px high and 128x128 px in-plane; in addition, the
    image can be rotated 360 degrees in each direction. The translation and
    rotation parameters will be 1/20 of these to allow proper visualisation of
    the transformations.
    '''
    fraction = 1/20
    ranges = {'k_trans': (-32 * fraction, 32 * fraction),
              'j_trans': (-128 * fraction, 128 * fraction),
              'i_trans': (-128 * fraction, 128 * fraction),
              'k_rot': (-180 * fraction, 180 * fraction),
              'j_rot': (-180 * fraction, 180 * fraction),
              'i_rot': (-180 * fraction, 180 * fraction)}

    parameters1 = randomise(ranges)
    parameters2 = randomise(ranges)
    parameters3 = randomise(ranges)

    print('Getting transformations...')
    t1 = RigidTransform(parameters1)
    _ = t1.compute_ddf(npyData.shape)
    t1_t2 = t1.compose(parameters2)
    t1_t2_t3 = t1_t2.compose(parameters3)
    t2 = RigidTransform(parameters2)
    _ = t2.compute_ddf(npyData.shape)
    t3 = RigidTransform(parameters3)
    _ = t3.compute_ddf(npyData.shape)

    print('Warping volumes...')
    vol1 = t1.warp(npyData)
    vol12 = t1_t2.warp(npyData)
    vol123 = t1_t2_t3.warp(npyData)
    vol1_2 = t2.warp(vol1)
    vol1_2_3 = t3.warp(vol1_2)

    print('Getting slices...')
    get_slices(main_data(vol1), '(Transformation T1)', save=True,
               save_label='t1')
    get_slices(main_data(vol12), '(Composed transformation T1⊕T2)', save=True,
               save_label='t1t2_composed')
    get_slices(main_data(vol123), '(Composed transformation T1⊕T2⊕T3)',
               save=True, save_label='t1t2t3_composed')
    get_slices(main_data(vol1_2), '(Sequential transformation T1→T2)',
               save=True, save_label='t1_t2_sequential')
    get_slices(main_data(vol1_2_3), '(Sequential transformation T1→T2→T3)',
               save=True, save_label='t1_t2_t3_sequential')

    '''
    Observations:
        It can be seen how the images from the rewarped volumes have lost
        information in contrast to the volumes warped only once with composed
        transformations: a blurring effect can be seen in the images
        transformed sequentially. This is due to the cumulative effect of
        repeated interpolation (data loss) in the rewarped images. In addition,
        the sequential transformations appear to warp the volume additionally
        compared to the composed ones.
    '''

    # EXPERIMENT 2
    print('\n' + '\u0332'.join('EXPERIMENT_2'))

    # Allowing the composing DDF flag
    print('Getting transformations...')
    t1.flag_composing_ddf = True
    t1_t2_flag = t1.compose(parameters2)
    t1_t2_flag.flag_composing_ddf = True
    t1_t2_t3_flag = t1_t2_flag.compose(parameters3)

    print('Warping volumes...')
    vol12_flag = t1_t2_flag.warp(npyData)
    vol123_flag = t1_t2_t3_flag.warp(npyData)

    print('Calculating differences...')
    diff_12 = ((vol12 - vol12_flag) ** 2) ** 0.5
    diff_mean_12 = diff_12.mean()
    diff_sd_12 = diff_12.std()

    diff_123 = ((vol123 - vol123_flag) ** 2) ** 0.5
    diff_mean_123 = diff_123.mean()
    diff_sd_123 = diff_123.std()

    print('Getting slices...')
    get_slices(main_data(vol12_flag), '[Composed transformation T1⊕T2 (after '
               f'enabling flag)\nDiff. mean: {diff_mean_12:.5g}; Diff. s.d.: '
               f'{diff_sd_12:.5g}]', save=True,
               save_label='t1t2_composed_flag')
    get_slices(main_data(vol123_flag), '[Composed transformation T1⊕T2⊕T3 '
               f'(after enabling flag)\nDiff. mean: {diff_mean_123:.5g}; '
               f'Diff. s.d.: {diff_sd_123:.5g}]', save=True,
               save_label='t1t2t3_composed_flag')

    """
    Observations:
        The mean and standard deviation of the intensity differences can be
        seen in the saved slices.
        Composing the DDF's using the newly implemented algorithm doesn't
        seem to affect the image quality (unlike the blur observed in the
        sequential case). However it still appears to warp the volume
        additionally compared to the original composing method. Visually, the
        degree of further warping seems to be equivalent to that seen in the
        sequential case. One could hypothesise that combining two
        transformations with the flag on produces a transformation equivalent
        to applying them sequentially, one after the other (but without the
        mentioned blurring).
    """
