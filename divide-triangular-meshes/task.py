from urllib.request import urlretrieve
import numpy as np
from skimage.measure import marching_cubes
from mesh import surface_dividing, visualise


def get_data():
    return np.load('3d_image.npy')


def between(minimum, maximum, number):
    return enumerate(np.linspace(minimum, maximum, number + 2)[1:-1], 1)


if __name__ == "__main__":
    sarData = get_data()
    mmVoxelDimensions = (2, 0.5, 0.5)

    sarTriangulatedSurface = marching_cubes(sarData, spacing=mmVoxelDimensions)
    sarFaces = sarTriangulatedSurface[0]
    rMin, rMax = sarFaces[:, 2].min(), sarFaces[:, 2].max()

    for i, sagittal_plane in between(rMin, rMax, 3):
        leftSurface, rightSurface = surface_dividing(sarTriangulatedSurface,
                                                     sagittal_plane)

        visualise(leftSurface[0], leftSurface[0], save=True, i=i,
                  info='_left')
        visualise(rightSurface[0], rightSurface[0], save=True, i=i,
                  info='_right')
        visualise(rightSurface[0], sarTriangulatedSurface, leftSurface[0],
                  save=True, i=i, info='_together')
