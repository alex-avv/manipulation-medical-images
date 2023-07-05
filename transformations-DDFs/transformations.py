from math import cos, sin, pi
import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt


class RigidTransform:
    def __init__(self, parameters=(0, 0, 0, 0, 0, 0)):
        """ Represents a 3D rigid transformation which can warp volumes.

        Parameters
        ----------
        parameters : length-6 tuple of floats, optional
            Transformation parameters, composed of 3 translations and 3
            rotations. The 1st, 2nd and 3rd instances represent the
            translations in the k-, j- and i-dim, respectively (in voxels).
            Similarly, the 4th, 5th and 6th instances depict the respective
            k-, j- and i-dim rotations (in degrees). The default is (0, 0, 0,
            0, 0, 0).

        Returns
        -------
        None.

        """
        k_trans, j_trans = -parameters[0], -parameters[1]
        i_trans, k_rot = -parameters[2], parameters[3]
        j_rot, i_rot = parameters[4], parameters[5]
        self.parameters = parameters

        # Translation vector
        self.Vtrans = np.array([[k_trans], [j_trans], [i_trans]])

        # Rotation matrix
        theta_k = k_rot * pi / 180
        theta_j = j_rot * pi / 180
        theta_i = i_rot * pi / 180
        Mrot_k = [[1, 0, 0],
                  [0, cos(theta_k), -sin(theta_k)],
                  [0, sin(theta_k), cos(theta_k)]]
        Mrot_j = [[cos(theta_j), 0, sin(theta_j)],
                  [0, 1, 0],
                  [-sin(theta_j), 0, cos(theta_j)]]
        Mrot_i = [[cos(theta_i), -sin(theta_i), 0],
                  [sin(theta_i), cos(theta_i), 0],
                  [0, 0, 1]]
        self.Mrot = np.dot(Mrot_k, np.dot(Mrot_j, Mrot_i))

        # Rigid transformation matrix
        self.Mrig = np.vstack([np.hstack([self.Mrot, self.Vtrans]),
                               [0, 0, 0, 1]])

        # Precomputing DDF and clearing flag_composing_ddf
        self.ddf = None
        self.flag_composing_ddf = False

    def __define_points_coors(self):
        """ Helper method to compute the transformation points and coordinates.
        """
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
        self.original_coors = np.concatenate((ZZ.reshape(-1, 1),
                                              YY.reshape(-1, 1),
                                              XX.reshape(-1, 1)), axis=1).T

    def __compute_warped_coordinates_from_M(self):
        """ Helper method to compute the coordinates after applying Mrig.
        """
        self.__define_points_coors()
        # Centering transformation
        center = self.len[0] // 2, self.len[1] // 2, self.len[2] // 2
        Mctr = np.eye(4)
        Mctr[0, 3] = 0 - center[0]
        Mctr[1, 3] = 0 - center[1]
        Mctr[2, 3] = 0 - center[2]
        # Inverse centering transformation
        Mctr_inv = np.eye(4)
        Mctr_inv[0, 3] = 0 + center[0]
        Mctr_inv[1, 3] = 0 + center[1]
        Mctr_inv[2, 3] = 0 + center[2]

        # Transformation matrix
        M = np.dot(Mctr_inv, np.dot(self.Mrig, Mctr))

        # Warped volume coordinates
        original_coors = np.concatenate((self.original_coors,
                                         np.ones((1, np.prod(self.len)))
                                         ))
        self.warped_coors = np.dot(M, original_coors)[:3]

    def __compute_warped_coordinates_from_DDF(self):
        """ Helper method to compute the coordinates after applying the DDF.
        """
        self.__define_points_coors()

        warped_coors = np.zeros((np.prod(self.len), 3))
        count = -1
        for k_slice in self.ddf:
            for j_row in k_slice:
                for vector_from_warped_to_original in j_row:
                    count += 1
                    warped_coors[count] = (self.original_coors.T[count] -
                                           vector_from_warped_to_original)
        self.warped_coors = warped_coors.T

    def compute_ddf(self, dim):
        """ Calculates the DDF of the transformation, given a volume size.

        Parameters
        ----------
        dim : length-3 tuple of floats
            Specifies the size of the volume to be warped.

        Returns
        -------
        3D ndarray object of ndarrays
            Numpy object representing the 3D displacement vector (from warped
            image to original image) at each (warped image) voxel locations.

        Coordinate System Notes
        -----------------------
        The coordinate system chosen is as follows:

        • The k-dim, (or 0-dim of the volume) is linked to the increasing
        superior anatomical coordinates. Similarly, the j-dim (or volume 1-dim)
        and i-dim (or volume 2-dim) are related to the increasing anterior and
        right coordinates.

        • The rotation of a 3d image about an axis is determined using the curl
        "right hand rule". If one was to look at the increasing axes from
        above, the +ve angles would correspond to the anticlockwise direction;
        on the contrary, the -ve angles would correspond to the clockwise
        direction.

        • The origin of an image volume is located at the array (0, 0, 0)
        position.

        • The unit used across all the dimensions is the voxel.

        """
        self.len = dim
        self.__compute_warped_coordinates_from_M()
        ddf_array_from_warped_to_original = (self.original_coors -
                                             self.warped_coors)

        ddf_from_warped_to_original = np.zeros(dim, dtype='object')
        row = -1
        for k, j, i in self.original_coors.T.astype('int'):
            row += 1
            (ddf_from_warped_to_original
             [k, j, i]) = ddf_array_from_warped_to_original.T[row]

        self.ddf = ddf_from_warped_to_original
        return ddf_from_warped_to_original

    def warp(self, data):
        """ Warpes a 3D image using the transfomation's DDF.

        Parameters
        ----------
        data : 3D ndarray of floats
            Numpy array representing a volume. The volume should respectively
            have superior, anterior, right coordinates for the array 0, 1, 2
            dimensions.

        Raises
        ------
        ValueError
            It is required to compute the DDF before warping the volume. Use
            RigidTransform.compute_ddf method.

        Returns
        -------
        data_interpn : 3D ndarray of floats
            Numpy array representing the warped volume.

        """
        if self.ddf is None:
            raise ValueError("It is required to compute the DDF before "
                             "warping the volume. Use "
                             "RigidTransform.compute_ddf method")
        self.__compute_warped_coordinates_from_DDF()

        # Interpolation
        data_interpn_flatten = interpn(self.points, data, self.warped_coors.T,
                                       bounds_error=False, fill_value=0)
        data_interpn = data_interpn_flatten.reshape(self.len)

        # Returning transformed volume
        return data_interpn

    def warp_extrap(self, data):
        """ Same as warp method, but extrapolating outlier values.

        Notes
        -----
        'fill_value' in scipy.interpolate.interpn is None instead of 0. This
        extrapolates values outside the interpolation domain during
        interpolation.

        """
        if self.ddf is None:
            raise ValueError("It is required to compute the DDF before "
                             "warping the volume. Use "
                             "RigidTransform.compute_ddf method")
        self.__compute_warped_coordinates_from_DDF()

        # Interpolation
        data_interpn_flatten = interpn(self.points, data, self.warped_coors.T,
                                       bounds_error=False, fill_value=None)
        data_interpn = data_interpn_flatten.reshape(self.len)

        # Returning transformed volume
        return data_interpn

    def compose(self, new_parameters=(0, 0, 0, 0, 0, 0)):
        """ Combines another rigid transformation into a new object.

        Uses the self transformation and a new transformation (represented by
        the new parameters) to create an equivalent transform.

        Parameters
        ----------
        new_parameters : length-6 tuple of floats, optional
            Second set of rigid transformation parameters. The default is (0,
            0, 0, 0, 0, 0).

        Returns
        -------
        composed_object : RigidTransform
            RigidTransform object representing the combined transform.

        Notes
        -----
        If 'flag_composing_ddf' member variable is True, the method uses a
        different algorithm to compute the composed DDF. By default, this flag
        is set to False when initiating an object instance.

        """
        compose_parameters = tuple(map(lambda a, b: a + b, self.parameters,
                                       new_parameters))
        if not self.flag_composing_ddf:
            composed_object = RigidTransform(compose_parameters)
            composed_object.compute_ddf(self.len)

            # Returning object with updated object members
            return composed_object

        else:
            # Constructing object with new parameters and getting its DDF
            new_object = RigidTransform(new_parameters)
            new_object.compute_ddf(self.len)
            new_ddf = new_object.ddf

            composed_ddf = self.composing_ddfs(self.ddf, new_ddf)

            composed_object = RigidTransform(compose_parameters)
            composed_object.len = self.len
            composed_object.ddf = composed_ddf

            # Returning object with updated object members
            return composed_object

    def composing_ddfs(self, ddf1, ddf2):
        """ Combines two DDFs, without using the composed Mrig.

        In order to fuse the DDFs, the algorithm initally resamples the first
        DDF into the second DDF's coordinate system. This is done by warping
        the first DDF with the transform defined by the second DDF. The
        resulting DDF is then added to the second DDF to get the combined DDF.

        Parameters
        ----------
        ddf1 : 3D ndarray object of ndarrays
            Numpy object representing a DDF [i.e. the 3D displacement vectors
            (from warped image to original image) at each (warped image) voxel
            location].
        ddf2 : 3D ndarray object of ndarrays
            Numpy object representing another DDF.

        Returns
        -------
        3D ndarray object of ndarrays
            Numpy object representing the combined DDF.

        """
        # Creating a local RigidTransform object to carry out ddf1's resampling
        # using ddf2's transformation
        object_ddf2 = RigidTransform()
        object_ddf2.len = ddf2.shape
        object_ddf2.ddf = ddf2

        # Separating and warping the vectors within ddf1
        empty_dff = np.zeros([ddf1.shape[0], ddf1.shape[1], ddf1.shape[2]])
        ddf1_k = np.copy(empty_dff)
        ddf1_j = np.copy(empty_dff)
        ddf1_i = np.copy(empty_dff)
        for k in range(ddf1.shape[0]):
            for j in range(ddf1.shape[1]):
                for i in range(ddf1.shape[2]):
                    ddf1_k[k, j, i] = ddf1[k, j, i][0]
                    ddf1_j[k, j, i] = ddf1[k, j, i][1]
                    ddf1_i[k, j, i] = ddf1[k, j, i][2]
        warped_ddf1_k = object_ddf2.warp_extrap(ddf1_k)
        warped_ddf1_j = object_ddf2.warp_extrap(ddf1_j)
        warped_ddf1_i = object_ddf2.warp_extrap(ddf1_i)

        # Combining ddf1's resampled vector components into a new 'warped' ddf1
        warped_ddf1 = np.copy(empty_dff).astype(dtype='object')
        for k in range(ddf1.shape[0]):
            for j in range(ddf1.shape[1]):
                for i in range(ddf1.shape[2]):
                    k_val = warped_ddf1_k[k, j, i]
                    j_val = warped_ddf1_j[k, j, i]
                    i_val = warped_ddf1_i[k, j, i]
                    warped_ddf1[k, j, i] = np.array([k_val, j_val, i_val])

        # Summing both DDFs (now in the same coordinate system) and returning
        # result
        return warped_ddf1 + ddf2


def between(minimum, maximum, number):
    return enumerate(np.linspace(minimum, maximum, number + 2)[1:-1]
                     .astype('int'), 1)


def get_slices(volume, more_info='', save=False, save_label=''):
    """ Plots 5 axial planes of a volume.

    Parameters
    ----------
    volume : 3D ndarray of floats
        Numpy array representing a volume. The volume should respectively
        have superior, anterior, right coordinates for the array 0, 1, 2
        dimensions.
    more_info : TYPE, optional
        Specifies additional information in the figure's title. The default is
        ''.
    save : bool, optional
        Specifies whether to save the plot as a PNG file. The default is False.
    save_label : str, optional
        Specifies label information in saved file's name. The default is ''.

    Returns
    -------
    None.

    """
    for i, axial_slice in between(0, volume.shape[0], 5):
        image = volume[axial_slice]
        # Normalising image between [0,1] for visualisation purposes
        image = (image - image.min()) / (image.max() - image.min())

        fig, ax = plt.subplots()
        plt.imshow(image, cmap='gray', origin='lower', vmin=0, vmax=1)
        ax.set_title(f'Axial slice of warped volume at Z{i}\n{more_info}')
        ax.set_ylabel('Anterior (px)')
        ax.set_xlabel('Right (px)')

        if save:
            plt.savefig(f'{save_label}_z{i}.png')
