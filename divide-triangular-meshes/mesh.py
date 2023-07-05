import numpy as np
import matplotlib.pyplot as plt


def surface_dividing(sarTriangulatedSurface, fSagittalPlane):
    """ Separates a triangulated surface by the specified sagittal plane.

    Parameters
    ----------
    sarTriangulatedSurface : length-4 tuple of ndarrays
        Triangulated surface as output by the skimage.measure marching_cubes
        function. The surface should respectively have superior, anterior,
        right coordinates for the surface M, N, P dimensions.
    iSagittalPlane : float
        Sagittal plane in right coordinates (or in N-dim).

    Returns
    -------
    list of length-4 tuple of ndarrays
        Left part of divided surface, given as a new triangulated surface
    list of length-4 tuple of ndarrays
        Right part of divided surface, given as a new triangulated surface

    """
    sarVerts, sarFaces, sarNormals, sarValues = sarTriangulatedSurface

    leftVerts, leftFaces = np.copy(sarVerts), np.copy(sarFaces)
    leftNormals, leftValues = np.copy(sarNormals), np.copy(sarValues)
    rightVerts, rightFaces = np.copy(sarVerts), np.copy(sarFaces)
    rightNormals, rightValues = np.copy(sarNormals), np.copy(sarValues)

    bRight = sarVerts[:, 2] > fSagittalPlane
    leftVerts[bRight] = ([leftVerts[:, 0].mean(), leftVerts[:, 1].mean(),
                          fSagittalPlane])
    leftNormals[bRight] = [0, 0, 0]
    leftValues[bRight] = 0
    leftSurface = (leftVerts, leftFaces, leftNormals, leftValues)

    bLeft = sarVerts[:, 2] < fSagittalPlane
    rightVerts[bLeft] = ([rightVerts[:, 0].mean(), rightVerts[:, 1].mean(),
                          fSagittalPlane])
    rightNormals[bLeft] = [0, 0, 0]
    rightValues[bLeft] = 0
    rightSurface = (rightVerts, rightFaces, rightNormals, rightValues)

    return [leftSurface], [rightSurface]


def visualise(sarSurface1, axisSurface, sarSurface2=None, save=False, i=0,
              info=''):
    """ Plots a triangulated surface in 3-D space.

    Parameters
    ----------
    sarSurface1 : length-4 tuple of ndarrays
        Triangulated surface (as output by the skimage.measure marching_cubes
        function) to be plotted.
    axisSurface : length-4 tuple of ndarrays
        Triangulated surface to determine the axes' length in the plot.
    sarSurface2 : length-4 tuple of ndarrays, optional
        Additional triangulated surface to be plotted in the same axes. The
        default is None.
    save : bool, optional
        Specifies whether to save the plot as a PNG file. The default is False.
    i : int, optional
        Specifies case number in saved PNG file's name. The default is 0.
    info : str, optional
        Specifies additional info in saved file's name. The default is ''.

    Returns
    -------
    None.

    """
    sarVerts1, sarFaces1, _, _ = sarSurface1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(sarVerts1[:, 1], sarVerts1[:, 2], sarVerts1[:, 0],
                    triangles=sarFaces1, color='b')

    if sarSurface2:
        sarVerts2, sarFaces2, _, _ = sarSurface2
        ax.plot_trisurf(sarVerts2[:, 1], sarVerts2[:, 2], sarVerts2[:, 0],
                        triangles=sarFaces2, color='c')

    ax.set_xlabel('Anterior [mm]')
    ax.set_ylabel('Right [mm]')
    ax.set_zlabel('Superior [mm]')
    axisVerts, _, _, _ = axisSurface
    ax.set_box_aspect((np.ptp(axisVerts[:, 1]), np.ptp(axisVerts[:, 2]),
                       np.ptp(axisVerts[:, 0])))  # 1:1:1 aspect ratio

    if save:
        fig.savefig(f'case{i}{info}.png')
    else:
        plt.show()
