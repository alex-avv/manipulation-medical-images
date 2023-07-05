import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter


class Image3D:
    def __init__(self, sar_data, vx_dim):
        """ Handles 3D medical images, allowing to resize them.

        Parameters
        ----------
        sar_data : 3D ndarray of floats
            Numpy array representing a volume. The volume should respectively
            have superior, anterior, right coordinates for the array 0, 1, 2
            dimensions.
        vx_dim : length-3 tuple of floats
            Voxel dimensions in the superior, anterior and right directions
            (in mm).

        Returns
        -------
        None.

        Coordinate System Used
        ----------------------
        The local image coordinate system of the Image3D class relates the
        k-dim (or 0-dim of the volume) to the superior anatomical coordinates.
        Similarly, it relates the j-dim (or volume 1-dim) and i-dim (or volume
        2-dim) to the anterior and right coordinates. The voxels in the train
        volume follow the defined dimensions, with the k-pixels representing 2
        mm in real space and the j- and i-pixels representing 0.5 mm. These
        voxel dimensions are updated every time the image is resized.

        """
        self.data = sar_data
        self.vx_dim = vx_dim
        self.__define_coords()

    def __define_coords(self):
        """ Helper method to compute the voxel coordinates.

        It also gets the voxel points and centering transformation.
        """
        self.len = self.data.shape
        k_len = self.len[0]
        j_len = self.len[1]
        i_len = self.len[2]
        k_points = np.arange(0, k_len)
        j_points = np.arange(0, j_len)
        i_points = np.arange(0, i_len)
        self.points = (k_points, j_points, i_points)

        ZZ, YY, XX = np.meshgrid(np.arange(0, k_len),
                                 np.arange(0, j_len),
                                 np.arange(0, i_len),
                                 indexing='ij')
        self.vol_coors = np.concatenate((ZZ.reshape(-1, 1), YY.reshape(-1, 1),
                                         XX.reshape(-1, 1),
                                         np.ones((np.prod(self.len), 1))),
                                        axis=1).T

        # Centering transformation
        self.center = k_len // 2, j_len // 2, i_len // 2
        centering_tf = np.eye(4)
        centering_tf[0, 3] = 0 - self.center[0]
        centering_tf[1, 3] = 0 - self.center[1]
        centering_tf[2, 3] = 0 - self.center[2]
        self.centering_tf = centering_tf
        # Inverse centering transformation
        centering_tf_inv = np.eye(4)
        centering_tf_inv[0, 3] = 0 + self.center[0]
        centering_tf_inv[1, 3] = 0 + self.center[1]
        centering_tf_inv[2, 3] = 0 + self.center[2]
        self.centering_tf_inv = centering_tf_inv

    def volume_resize(self, resize_ratio=(1, 1, 1)):
        """ Resizes the 3D image within the object by the specified ratio.

        Parameters
        ----------
        resize_ratio : length-3 tuple of floats, optional
            Resize ratio, in the superior, anterior and right dimensions. The
            default is (1, 1, 1).

        Returns
        -------
        Image3D
            Image3D object representing the resized 3D image.

        """
        # If any resize ratio is > 1, padding data and re-computing voxel
        # coordinates
        padz, pady, padx = 0, 0, 0
        if resize_ratio[0] > 1:
            padz = int((self.len[0] * (resize_ratio[0] - 1)) / 2)
        if resize_ratio[1] > 1:
            pady = int((self.len[1] * (resize_ratio[1] - 1)) / 2)
        if resize_ratio[2] > 1:
            padx = int((self.len[2] * (resize_ratio[2] - 1)) / 2)

        self_data_copy = np.copy(self.data)  # Copying data to reset it after
        # transformation
        self.data = np.pad(self.data, [(padz, padz), (pady, pady),
                                       (padx, padx)])
        self.__define_coords()

        # Updating voxel dimensions
        vx_dim = tuple(map(lambda a, b: a / b, self.vx_dim, resize_ratio))

        # Scaling matrix
        Msc = np.array([[1 / resize_ratio[0], 0, 0, 0],
                        [0, 1 / resize_ratio[1], 0, 0],
                        [0, 0, 1 / resize_ratio[2], 0],
                        [0, 0, 0, 1]])

        # Transformation matrix
        M = np.dot(self.centering_tf_inv, np.dot(Msc, self.centering_tf))

        # Interpolation parameters
        xi = np.dot(M, self.vol_coors)[:3].T  # New coordinates from matrix
        # multiplication

        # Interpolation
        data_interpn_flatten = interpn(self.points, self.data, xi,
                                       bounds_error=False, fill_value=0)
        data_interpn = data_interpn_flatten.reshape(self.len)

        # If any resize ratio is < 1, cropping data
        cropz, cropy, cropx = 0, 0, 0
        if resize_ratio[0] < 1:
            cropz = int((data_interpn.shape[0] * (1 - resize_ratio[0])) / 2)
        if resize_ratio[1] < 1:
            cropy = int((data_interpn.shape[1] * (1 - resize_ratio[1])) / 2)
        if resize_ratio[2] < 1:
            cropx = int((data_interpn.shape[2] * (1 - resize_ratio[2])) / 2)
        data_interpn = data_interpn[cropz:data_interpn.shape[0] - cropz,
                                    cropy:data_interpn.shape[1] - cropy,
                                    cropx:data_interpn.shape[2] - cropx]

        # Setting object attributes to their original values
        self.data = self_data_copy
        self.__define_coords()

        # Returning transformed object
        return Image3D(data_interpn, vx_dim)

    def volume_resize_antialias(self, resize_ratio=(1, 1, 1),
                                sd=(0, 0, 0)):
        """ Applies a Gaussian filter before resizing the object 3D image.

        Parameters
        ----------
        resize_ratio : length-3 tuple of floats, optional
            Resize ratio, in the superior, anterior and right dimensions. The
            default is (1, 1, 1).
        sd : length-3 tuple of floats, optional
            Standard deviation of the Gaussian filter, in the superior,
            anterior and right dimensions (in mm). The default is (0, 0, 0).

        Returns
        -------
        Image3D
            Image3D object representing the resized 3D image.

        """
        # Updating sd to match volume coordinate system
        sd = tuple(map(lambda a, b: a / b, sd, self.vx_dim))

        image_3d = Image3D(self.data, self.vx_dim)
        image_3d.data = gaussian_filter(image_3d.data, sigma=sd)
        return image_3d.volume_resize(resize_ratio)


def between(minimum, maximum, number):
    return enumerate(np.linspace(minimum, maximum, number + 2)[1:-1]
                     .astype('int'), 1)


def k_boundaries(volume):
    k_min = np.any(volume, axis=(1, 2)).argmax()
    k_max = volume.shape[0] - np.flip(np.any(volume, axis=(1, 2))).argmax() - 1
    return k_min, k_max


def get_slices(volume, vx_dim, filtering, m, n=None, timing=None,
               more_info=None, save=False):
    """ Plots 5 axial planes of a volume, showing relevant information.

    Parameters
    ----------
    volume : 3D ndarray of floats
        Numpy array representing a volume. The volume should respectively
        have superior, anterior, right coordinates for the array 0, 1, 2
        dimensions.
    vx_dim : length-3 tuple of floats
        Voxel dimensions in the superior, anterior and right directions
        (in mm).
    filtering : bool
        Specifies whether filtering was used to get the given volume.
    m : int
        Specifies experiment number.
    n : int, optional
        Specifies case number. The default is None.
    timing : float, optional
        Specifies execution time (in s) in the figure's title. The default is
        None.
    more_info : length-2 tuple of floats, optional
        Specifies mean and standard deviation of voxel-level intensity
        differences in the figure's title. The default is None.
    save : bool, optional
        Specifies whether to save the plot as a PNG file. The default is False.

    Returns
    -------
    None.

    """

    filter_info = 'Filtered' if filtering else 'Non-filtered'
    scenario = f', Scen. {n}' if n else ''
    timing_info = ('Execution time: ' + str(round(timing * 1e3)) + ' ms'
                   if timing else '')
    more_info = (f'Diff. mean: {more_info[0]}\n'
                 f'Diff. s.d.: {more_info[1]}'
                 if more_info else '')
    text1 = f'_scenario{n}' if n else ''
    text2 = 'with_filter' if filtering else 'without_filter'

    k_min, k_max = k_boundaries(volume)
    for i, axial_slice in between(k_min, k_max, 5):
        image = volume[axial_slice]
        # Normalising image between [0,1] for visualisation purposes
        image = (image - image.min()) / (image.max() - image.min())

        fig, ax = plt.subplots()
        plt.imshow(image, cmap='gray', origin='lower', vmin=0, vmax=1)

        ax.set_title(f'Exp. {m}{scenario}: Axial slice of resized volume at '
                     f'Z{i} ({filter_info}).\n'
                     f'{timing_info}{more_info}')
        ax.set_ylabel(f'Anterior [{vx_dim[1]:.2g} mm]')
        ax.set_xlabel(f'Right [{vx_dim[2]:.2g} mm]')

        if save:
            fig.savefig(f'exp{m}{text1}_z{i}_{text2}.png')
