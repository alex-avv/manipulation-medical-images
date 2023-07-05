from math import cos, sin, pi
import numpy as np
from numpy import sqrt, ogrid, exp
from scipy.interpolate import interpn
import matplotlib.pyplot as plt


def k_slice_coords(image):
    """ Helper function to get the 3D coordinates of a k-dim slice.
    """
    XX, YY = np.meshgrid(np.arange(-image.shape[1] // 2,
                                   image.shape[1] // 2),
                         np.arange(-image.shape[0] // 2,
                                   image.shape[0] // 2))
    ZZ = np.full(image.shape, 0, dtype=int)
    return XX, YY, ZZ


def j_slice_coords(image):
    """ Helper function to get the 3D coordinates of a j-dim slice.
    """
    XX, ZZ = np.meshgrid(np.arange(-image.shape[1] // 2,
                                   image.shape[1] // 2),
                         np.arange(-image.shape[0] // 2,
                                   image.shape[0] // 2))
    YY = np.full(image.shape, 0, dtype=int)
    return XX, YY, ZZ


def i_slice_coords(image):
    """ Helper function to get the 3D coordinates of an i-dim slice.
    """
    YY, ZZ = np.meshgrid(np.arange(-image.shape[1] // 2,
                                   image.shape[1] // 2),
                         np.arange(-image.shape[0] // 2,
                                   image.shape[0] // 2))
    XX = np.full(image.shape, 0, dtype=int)
    return XX, YY, ZZ


def show_slice(image, axis='', trans='', x_rot='', y_rot='',
               z_rot='', filtered='', visualisation='', volume=None,
               resol_3d=False, save=False):
    """ Visualises an image volume slice in 2D or 3D.

    This function uses the parameters as used by the slice_at function to
    obtain the image slice.

    Parameters
    ----------
    image : 2D ndarray of floats
        Numpy array representing the sliced image.
    axis : str, optional
        Dimension at which slice was retrieved (either x, y or z). The default
        is ''.
    trans : float, optional
        Location at which slice was retrieved (in px). The default is ''.
    x_rot : float, optional
        Slice angle in the x-dim (in degrees). The default is ''.
    y_rot : float, optional
        Slice angle in the y-dim (in degrees). The default is ''.
    z_rot : float, optional
        Slice angle in the z-dim (in degrees). The default is ''.
    filtered : str, optional
        Specifies the type of filtering used in the image. The default is ''.
    visualisation : str, optional
        Type of visualisation to be shown (either 2d or 3d). The default is ''.
    volume : 3D ndarray of floats, optional
        Numpy array representing the image volume from which the slice was
        obtained.
    resol_3d : float, optional
        Resolution of the image slices in the 3D plot. The default is False.
    save : bool, optional
        Specifies whether to save the plot as a PNG file. The default is False.

    Returns
    -------
    None.

    Notes
    -----
    For a faster execution of the function in 3D, set resol_3d to False. This
    runs the matplotlib.pyplot.figure().add_subplot(projection='3d')
    .plot_surface method with its default resolution (fast).

    """

    if visualisation not in ['2d', '3d']:
        raise ValueError("Visualisation must be specified either as '2d' or "
                         "'3d'")
    if visualisation == '3d' and volume is None:
        raise ValueError("An image volume must be passed for 3D "
                         "visualisation, set volume in function arguments")

    # ~~~ TITLE AND SAVE TAG ~~~
    # Descriptive information in figure title and label if saving file
    if visualisation == '2d':
        save_info = 'slice'
    elif visualisation == '3d':
        save_info = 'slices'

    if axis == '':
        title_info = 'Slice at N.S. = '
        save_info += '_ns'
    else:
        title_info = f'Slice at {axis.upper()} = '
        save_info += f'_{axis}'
    if trans == '':
        title_info += 'N.S. mm'
        save_info += 'ns'
    else:
        title_info += f'{trans} mm'
        save_info += f'{trans}'

    if x_rot or y_rot or z_rot:
        save_info += '_angle'
        if x_rot:
            save_info += f'_x{round(x_rot)}'
            if not y_rot and not z_rot:
                title_info += f' and θ(X) = {x_rot}°'
            else:
                title_info += f', θ(X) = {x_rot}°'
        if y_rot:
            save_info += f'_y{round(y_rot)}'
            if not z_rot:
                title_info += f' and θ(Y) = {y_rot}°'
            else:
                title_info += f', θ(Y) = {y_rot}°'
        if z_rot:
            save_info += f'_z{round(z_rot)}'
            title_info += f' and θ(Z) = {z_rot}°'

    if filtered == 'no':
        title_info += '\n(Not filtered)'
        save_info += '_not_fld'
    elif filtered == '2d':
        title_info += '\n(2D filtered)'
        save_info += '_2d_fld'
    elif filtered == '3d':
        title_info += '\n(3D filtered)'
        save_info += '_3d_fld'
    elif not filtered:
        title_info += '\n(Filtering N.S.)'
        save_info += '_flrg_ns'

    # ~~~ 2D PLOTTING ~~~
    if visualisation == '2d':
        # Normalising image between [0,1] for visualisation purposes
        image = (image - image.min()) / (image.max() - image.min())

        fig, ax = plt.subplots()
        plt.imshow(image, cmap='gray', origin='lower', vmin=0, vmax=1)

    # ~~~ 3D PLOTTING ~~~
    elif visualisation == '3d':
        if (axis == '' or trans == '' or x_rot == '' or y_rot == ''
                or z_rot == ''):
            raise ValueError("Parameters used in transformation must be "
                             "specified")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')  # computed_zorder=False

        def split_into_4(image):
            """ Helper function to split an 2D ndarray into four parts.

            This aids in the 3D visualisation of all the slices.
            """
            image_1j, image_2j = np.array_split(image, 2, axis=1)
            image_1j_1i, image_1j_2i = np.array_split(image_1j, 2, axis=0)
            image_2j_1i, image_2j_2i = np.array_split(image_2j, 2, axis=0)
            return image_1j_1i, image_2j_1i, image_1j_2i, image_2j_2i

        def show_in_3d(a_slice, x, y, z, resolution=False, split=True):
            """ Plots an image volume slice in 3D.

            Parameters
            ----------
            a_slice : 2D ndarray of floats
                Numpy array representing the sliced image.
            x : 2D ndarray of ints
                Numpy array representing the 3D x-coordinates.
            y : 2D ndarray of ints
                Numpy array representing the 3D y-coordinates.
            z : 2D ndarray of ints
                Numpy array representing the 3D z-coordinates..
            resolution : float, optional
                Resolution of the image slices in the 3D plot. The default is
                False.
            split : bool, optional
                Specifies whether to split the image into four parts when
                plotting. The default is True.

            Returns
            -------
            None.

            """
            if split:
                slices, xs = split_into_4(a_slice), split_into_4(x),
                ys, zs = split_into_4(y), split_into_4(z)

                for quarter_slice, x4, y4, z4 in zip(slices, xs, ys, zs):
                    if resolution:
                        ax.plot_surface(x4, y4, z4,
                                        rstride=resolution, cstride=resolution,
                                        facecolors=plt.cm.gray(quarter_slice))
                    else:
                        ax.plot_surface(x4, y4, z4,
                                        facecolors=plt.cm.gray(quarter_slice))
            elif not split:
                if resolution:
                    ax.plot_surface(x, y, z,
                                    rstride=resolution, cstride=resolution,
                                    facecolors=plt.cm.gray(a_slice))
                else:
                    ax.plot_surface(x, y, z, facecolors=plt.cm.gray(a_slice))

        # Converting the transfomation angles to radians
        theta_z, theta_y = z_rot * pi / 180, y_rot * pi / 180
        theta_x = x_rot * pi / 180

        # Getting rotation matrix to rotate the slice in 3D
        Mrot_z = [[1, 0, 0],
                  [0, cos(theta_z), -sin(theta_z)],
                  [0, sin(theta_z), cos(theta_z)]]
        Mrot_y = [[cos(theta_y), 0, sin(theta_y)],
                  [0, 1, 0],
                  [-sin(theta_y), 0, cos(theta_y)]]
        Mrot_x = [[cos(theta_x), -sin(theta_x), 0],
                  [sin(theta_x), cos(theta_x), 0],
                  [0, 0, 1]]
        Mrot = np.dot(Mrot_z, np.dot(Mrot_y, Mrot_x))

        # Warping the slice
        a_slice = image.astype('int')  # Changing to integer type for adequate
        # 3D visualisation with matplotlib.pyplot.figure().add_subplot(
        # projection='3d').plot_surface method
        if axis == 'z':
            XX, YY, ZZ = k_slice_coords(a_slice)
            x = XX * Mrot[2, 2] + YY * Mrot[2, 1] + ZZ * Mrot[2, 0]
            y = XX * Mrot[1, 2] + YY * Mrot[1, 1] + ZZ * Mrot[1, 0]
            z = XX * Mrot[0, 2] + YY * Mrot[0, 1] + ZZ * Mrot[0, 0] + trans

        elif axis == 'y':
            XX, YY, ZZ = j_slice_coords(a_slice)
            x = XX * Mrot[2, 2] + YY * Mrot[2, 1] + ZZ * Mrot[2, 0]
            y = XX * Mrot[1, 2] + YY * Mrot[1, 1] + ZZ * Mrot[1, 0] + trans
            z = XX * Mrot[0, 2] + YY * Mrot[0, 1] + ZZ * Mrot[0, 0]

        elif axis == 'x':
            XX, YY, ZZ = i_slice_coords(a_slice)
            x = XX * Mrot[2, 2] + YY * Mrot[2, 1] + ZZ * Mrot[2, 0] + trans
            y = XX * Mrot[1, 2] + YY * Mrot[1, 1] + ZZ * Mrot[1, 0]
            z = XX * Mrot[0, 2] + YY * Mrot[0, 1] + ZZ * Mrot[0, 0]

        # If any parts of the slice are outside the volume frame, warping
        # them to the volume edges. These points will always be equal to 0 as
        # during interpolation outlier values are resetted. This was done for
        # a neater plot and for easier visualisation.
        x[x > volume.shape[2] // 2] = volume.shape[2] // 2
        x[x < -volume.shape[2] // 2] = -volume.shape[2] // 2
        y[y > volume.shape[1] // 2] = volume.shape[1] // 2
        y[y < -volume.shape[1] // 2] = -volume.shape[1] // 2
        z[z > volume.shape[0] // 2] = volume.shape[0] // 2
        z[z < -volume.shape[0] // 2] = -volume.shape[0] // 2

        # Plotting transformed slice
        show_in_3d(a_slice, x, y, z, resolution=resol_3d)

        # Plotting axial, coronal and sagittal slices at centre of volume
        axial_slice = volume[volume.shape[0] // 2]
        XX, YY, ZZ = k_slice_coords(axial_slice)
        show_in_3d(axial_slice, XX, YY, ZZ, resolution=resol_3d)
        coronal_slice = volume[:, volume.shape[1] // 2, :]
        XX, YY, ZZ = j_slice_coords(coronal_slice)
        show_in_3d(coronal_slice, XX, YY, ZZ, resolution=resol_3d)
        sagittal_slice = volume[..., volume.shape[2] // 2]
        XX, YY, ZZ = i_slice_coords(sagittal_slice)
        show_in_3d(sagittal_slice, XX, YY, ZZ, resolution=resol_3d)

        ax.set_xlabel('Left [0.2 mm]')
        ax.set_ylabel('Anterior [0.2 mm]')
        ax.set_zlabel('Superior [2 mm]')
        ax.invert_xaxis()

    ax.set_title(f'{title_info}')
    # ax.view_init(elev=30, azim=60)
    if save:
        # print(f'{save_info}.png')
        plt.savefig(f'{save_info}.png')
    else:
        plt.show()


def slice_at(sal_data, axis='z', trans=0, x_rot=0, y_rot=0, z_rot=0,
             kernel=False):
    ''' Gets a slice from an image volume at the specified location and angle.

    Parameters
    ----------
    sal_data : 3D ndarray of floats
        Numpy array representing a volume. The volume should respectively
        have superior, anterior, left coordinates for the array 0, 1, 2
        dimensions.
    axis : str, optional
        Dimension at which to retrieve the slice (either x, y or z). The
        default is 'z'.
    trans : float, optional
        Location at which to get the slice (in px): this is given as a
        translation parameter. The default is 0.
    x_rot : float, optional
        Angle in the x-dim of the slice (in degrees): this is given as a
        rotation parameter. The default is 0.
    y_rot : float, optional
        Angle in the y-dim of the slice (in degrees): this is given as a
        rotation parameter. The default is 0.
    z_rot : float, optional
        Angle in the z-dim of the slice (in degrees): this is given as a
        rotation parameter. The default is 0.
    kernel : bool, optional
        Specifies whether the re-slicing is being applied to a filter kernel or
        not. The default is False.

    Returns
    -------
    2D ndarray of floats
        Numpy array representing the sliced image.

    Coordinate System Notes
    -----------------------
        The coordinate system chosen is as follows:

        • The x, y, z dimensions represent the volume 2, 1, 0 dimensions,
        respectively.

        • The rotation of a 3d image about an axis is determined using the curl
        "right hand rule". If one was to look at the increasing axes from
        above, the +ve angles would correspond to the anticlockwise direction;
        on the contrary, the -ve angles would correspond to the clockwise
        direction.

        • The origin of an image volume is located at the array central
        position. For example, if the array is 3×9×5, the origin would be at
        (2, 5, 3).

        • The unit used across all the dimensions is the voxel.

    '''

    if axis not in ['x', 'y', 'z']:
        raise ValueError("Selected axis must be either 'x', 'y' or 'z'")
    elif axis == 'x':
        x_trans, y_trans, z_trans = trans, 0, 0
    elif axis == 'y':
        x_trans, y_trans, z_trans = 0, trans, 0
    elif axis == 'z':
        x_trans, y_trans, z_trans = 0, 0, trans

    if (x_rot > 45 or x_rot <= -45 or y_rot > 45 or y_rot <= -45 or
            z_rot > 45 or z_rot <= -45):
        raise ValueError("Chosen angles must all be between -45° and 45°")

    # Calculating the maximum length that could be occupied by the rotation
    max_len = np.ceil(sqrt(sal_data.shape[0] ** 2 + sal_data.shape[1] ** 2
                           + sal_data.shape[2] ** 2))
    max_len_yz = np.ceil(sqrt(sal_data.shape[0] ** 2 + sal_data.shape[1] ** 2))
    max_len_xz = np.ceil(sqrt(sal_data.shape[0] ** 2 + sal_data.shape[2] ** 2))

    if not kernel:
        # Custom data padding for the experiment
        pad_x = np.ceil((max_len_xz - sal_data.shape[2]) / 2).astype('int')
        pad_y = np.ceil((max_len_yz - sal_data.shape[1]) / 2).astype('int')
        # pad_z = np.ceil(sal_data.shape[0] / 10).astype('int')
        pad_z = 0
        data = np.pad(sal_data, [(pad_z, pad_z), (pad_y, pad_y),
                                 (pad_x, pad_x)])
    else:
        # Padding data so no information is lost
        pad_x = np.ceil((max_len - sal_data.shape[2]) / 2).astype('int')
        pad_y = np.ceil((max_len - sal_data.shape[1]) / 2).astype('int')
        pad_z = np.ceil((max_len - sal_data.shape[0]) / 2).astype('int')
        data = np.pad(sal_data, [(pad_z, pad_z), (pad_y, pad_y),
                                 (pad_x, pad_x)])

    # ~~~ TRANSFORMATION MATRIX ~~~
    # Translation vector
    Vtrans = np.array([[z_trans], [y_trans], [x_trans]])

    # From degrees to radians
    theta_z, theta_y = z_rot * pi / 180, y_rot * pi / 180
    theta_x = x_rot * pi / 180

    # Rotation matrix
    Mrot_z = [[1, 0, 0],
              [0, cos(theta_z), -sin(theta_z)],
              [0, sin(theta_z), cos(theta_z)]]
    Mrot_y = [[cos(theta_y), 0, sin(theta_y)],
              [0, 1, 0],
              [-sin(theta_y), 0, cos(theta_y)]]
    Mrot_x = [[cos(theta_x), -sin(theta_x), 0],
              [sin(theta_x), cos(theta_x), 0],
              [0, 0, 1]]
    Mrot = np.dot(Mrot_z, np.dot(Mrot_y, Mrot_x))

    # Rigid transformation matrix
    Mrig = np.vstack([np.hstack([Mrot, Vtrans]), [0, 0, 0, 1]])

    # Centering transformation
    center = data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2
    Mctr = np.eye(4)
    Mctr[0, 3] = 0 - center[0]
    Mctr[1, 3] = 0 - center[1]
    Mctr[2, 3] = 0 - center[2]
    # Inverse centering transformation
    Mctr_inv = np.eye(4)
    Mctr_inv[0, 3] = 0 + center[0]
    Mctr_inv[1, 3] = 0 + center[1]
    Mctr_inv[2, 3] = 0 + center[2]

    M = np.dot(Mctr_inv, np.dot(Mrig, Mctr))

    # ~~~ INTERPOLATION ~~~
    # Getting original volume coordinates before interpolation
    z_len, y_len, x_len = data.shape[0], data.shape[1], data.shape[2]
    z_points = np.arange(0, z_len)
    y_points = np.arange(0, y_len)
    x_points = np.arange(0, x_len)
    points = (z_points, y_points, x_points)

    ZZ, YY, XX = np.meshgrid(np.arange(0, z_len), np.arange(0, y_len),
                             np.arange(0, x_len), indexing='ij')

    original_coors = np.concatenate((ZZ.reshape(-1, 1), YY.reshape(-1, 1),
                                     XX.reshape(-1, 1)), axis=1).T

    # Warped volume coordinates
    original_coors_pad = np.concatenate((original_coors,
                                         np.ones((1, np.prod(data.shape)))
                                         ))
    warped_coors = np.dot(M, original_coors_pad)[:3]

    data_interpn_flatten = interpn(points, data, warped_coors.T,
                                   bounds_error=False, fill_value=0)
    data_interpn = data_interpn_flatten.reshape(data.shape)

    # ~~~ VISUALISATION ~~~
    def get_image(data, axis, trans):
        if axis == 'x':
            idx = 2
        elif axis == 'y':
            idx = 1
        elif axis == 'z':
            idx = 0
        if trans < -data.shape[idx] // 2 or trans > (data.shape[idx] - 1) // 2:
            raise ValueError("Selected location of the slice is larger than "
                             "the volume")

        if axis == 'x':
            return data[..., data.shape[idx] // 2]
        elif axis == 'y':
            return data[:, data.shape[idx] // 2, :]
        if axis == 'z':
            return data[data.shape[idx] // 2]

    image = get_image(data_interpn, axis, trans)
    return image


def gaussian_2d(sigma_d, size=(3, 3)):
    # Creating a 2D grid of distances from the center
    x, y = (size[1] - 1) // 2, (size[0] - 1) // 2
    dist_1d = np.array(ogrid[-y:y + 1, -x:x + 1], dtype='object')

    # Finding the combined x- and y-dim squared distance at each pixel
    dist_sq = (dist_1d ** 2)[1] + (dist_1d ** 2)[0]

    # Building the 2D gaussian
    gaussian = (1 / (2 * pi *
                     sigma_d ** 2)) * exp((-1 / 2) * (dist_sq / sigma_d ** 2))

    # Normalisation
    gaussian = gaussian / gaussian.sum()
    return gaussian


def gaussian_3d(sigma_d, size=(3, 3, 3)):
    # Creating a 3D grid of distances from the center
    x, y, z = (size[2] - 1) // 2, (size[1] - 1) // 2, (size[0] - 1) // 2
    dist_1d = np.array(ogrid[-z:z + 1, -y:y + 1, -x:x + 1], dtype='object')

    # Finding the combined x-, y- and z-dim distance at each voxel
    dist_sq = (dist_1d ** 2)[2] + (dist_1d ** 2)[1] + (dist_1d ** 2)[0]

    # Building the 3D gaussian
    gaussian = (1 / (2 * pi *
                     sigma_d ** 2)) * exp((-1 / 2) * (dist_sq / sigma_d ** 2))

    # Normalisation
    gaussian = gaussian / gaussian.sum()
    return gaussian


def bilateral_filter(data, sigma_r, gaussian):
    # Getting the intensity at the centre of the kernel
    centre_i = gaussian.shape[0] // 2
    centre = data[centre_i]

    # Finding the similarity value at each pixel/voxel
    diff = data - centre
    diff_sq = diff ** 2

    # Building the intensity kernel
    intensity = exp((-1 / 2) * (diff_sq / sigma_r ** 2))

    bilateral = intensity * gaussian
    # Normalised filtered data
    data_filtered = np.sum(data * bilateral) / np.sum(bilateral)

    return data_filtered


# ~~~ TESTING CODE ~~~
# if __name__ == "__main__":

    # import cv2
    # from scipy import ndimage
    # img = cv2.imread('kitten.jpg')
    # img = img.mean(axis=2).astype('uint8')
    # vol = np.stack((img, img, img))
    # sigma_d = 0
    # sigma_r = 30

    ## 2D filtering
    # kernel_size = (21, 21)
    # gaussian_kernel = gaussian_2d(sigma_r, kernel_size)
    # gaussian_kernel_flatten = np.ravel(gaussian_kernel)
    # print('Filtering...')
    # kwd_args = dict(sigma_r=sigma_r, gaussian=gaussian_kernel_flatten)
    # img_filtered = ndimage.generic_filter(img, bilateral_filter, kernel_size,
    #                                       extra_keywords=kwd_args)
    # plt.subplots()
    # plt.imshow(img_filtered, cmap='gray')

    ### Testing 2D filtering with padded kernel gives same results
    ## gaussian_kernel_pad = np.vstack([np.zeros([1, 23]),
    ##                                 np.hstack([np.zeros([21, 1]),
    ##                                            gaussian_kernel,
    ##                                            np.zeros([21, 1])]),
    ##                                 np.zeros([1, 23])])
    ## gaussian_kernel_pad_flatten = np.ravel(gaussian_kernel_pad)
    ## print('Filtering...')
    ## kwd_args = dict(sigma_r=sigma_r, gaussian=gaussian_kernel_pad_flatten)
    ## img_filtered_pad_gauss = ndimage.generic_filter(img, bilateral_filter,
    ##                                               gaussian_kernel_pad.shape,
    ##                                               extra_keywords=kwd_args)
    ## np.testing.assert_equal(img_filtered, img_filtered_pad_gauss)

    ## 3D filtering
    # kernel_size = (3, 21, 21)
    # gaussian_kernel = gaussian_3d(sigma_r, kernel_size)
    # gaussian_kernel_flatten = np.ravel(gaussian_kernel)
    # print('Filtering...')
    # kwd_args = dict(sigma_r = sigma_r, gaussian = gaussian_kernel_flatten)
    # vol_filtered = ndimage.generic_filter(vol, bilateral_filter, kernel_size,
    #                                       extra_keywords=kwd_args)
    # plt.subplots()
    # plt.imshow(vol_filtered[0], cmap='gray')
