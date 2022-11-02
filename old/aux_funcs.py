import numpy as np
from matplotlib import pyplot as plt
from skimage import color
from skimage.transform import rescale
from skimage.draw import disk as circle, polygon
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.exposure import equalize_adapthist
from skimage.feature import canny
from skimage.filters import difference_of_gaussians, median, apply_hysteresis_threshold
from skimage.morphology import binary_closing, disk, binary_erosion, binary_dilation
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import active_contour
from scipy.ndimage import binary_fill_holes


mold_long_edge = 7
mold_short_edge = 5
mold_diag = np.sqrt(mold_long_edge**2 + mold_short_edge**2)


def rgb2rgba(image: np.ndarray):
    return (np.concatenate([np.flipud(image), np.ones((*image.shape[:-1], 1))], axis=-1)*255).astype(np.uint8)


def read_im(im_path: str, scale: float=1, match_size: tuple=None):
    """
    Read an image and resize it to the correct scale/size
    :param im_path: the path to the image that should be read
    :param scale: the amount the image should be rescaled
    :param match_size: if given, the image will be reshaped to match the given size
    :return: the scaled image
    """
    image = plt.imread(im_path)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # if scale > 1: scale = scale/np.max(image.shape)
    if match_size is None: match_size = [int(np.max(image.shape)*scale), int(np.max(image.shape)*scale)]
    scale = match_size[np.argmax(image.shape)]/image.shape[np.argmax(image.shape)]
    image = rescale(color.rgb2gray(image) if im_path.endswith('jpg') else image, scale)
    image = np.pad(image, [[0, match_size[0]-image.shape[0]], [0, match_size[1]-image.shape[1]]])
    return color.gray2rgb(image)


def thresh_find_cell(w: np.ndarray, circ, low_quant: float, high_quant: float, erosion_rad: int, max_cells: int):
    if low_quant > 1: low_quant = low_quant/100
    if high_quant > 1: high_quant = high_quant/100
    if low_quant > high_quant: low_quant = high_quant

    well_int = np.quantile(w[circ], q=.75) * np.ones(w.shape)
    well_int[circ] = w[circ]
    well_int[well_int > np.quantile(w[circ], q=.75)] = np.quantile(w[circ], q=.75)
    well_int = (well_int - np.min(well_int)) / (np.max(well_int) - np.min(well_int))
    well_int = well_int[:, :, 0]
    # well_int = equalize_adapthist(median(well_int, disk(5)))
    well_int = median(well_int, disk(2))

    well = apply_hysteresis_threshold(1 - well_int, low=np.quantile(1 - well_int, q=low_quant),
                                      high=np.quantile(1 - well_int, q=high_quant))
    well = well.astype(int)
    well = binary_erosion(binary_closing(well, disk(erosion_rad)), disk(max(erosion_rad-1, 1)))
    well = binary_fill_holes(well)

    # find labels of regions
    labels, n_labs = label(well, return_num=True, connectivity=1)
    props = [p for p in regionprops(labels) if p.area > 70 and p.eccentricity < .8]
    areas = [p.area for p in props]
    inds = np.argsort(areas)
    props = [props[i] for i in inds[-max_cells:]]

    # remove regions smaller than the top areas
    cell = np.zeros(well.shape, dtype=int)
    for p in props:
        cell[p.coords[:, 0], p.coords[:, 1]] = 1

    return cell, props


def active_find_cell(w: np.ndarray, circ, quant: float=.8, erosion_rad: int=4, max_cells: int=1,
                     alpha: float=.5, beta: float=1, gamma: float=.01):
    if quant > 1: quant = quant/100
    # pre-process well image to remove the well edge
    well_int = np.quantile(w[circ], q=.75) * np.ones(w.shape)
    well_int[circ] = w[circ]
    well_int = (well_int - np.min(well_int)) / (np.max(well_int) - np.min(well_int))
    well = well_int[:, :, 0]

    # threshold image to find crude approximation of cells
    b = median(well, disk(2))
    b = apply_hysteresis_threshold(1 - b, low=np.quantile(1 - well_int, q=quant),
                                   high=np.quantile(1 - well_int, q=quant))
    b = b.astype(int)
    b = binary_erosion(binary_closing(b, disk(erosion_rad)), disk(max(erosion_rad - 1, 1)))
    b = binary_dilation(b, disk(10))

    # find regions
    labels, n_labs = label(b, return_num=True, connectivity=1)
    props = [p for p in regionprops(labels) if p.area > 70 and p.eccentricity < .8]
    areas = [p.area for p in props]
    srt = np.argsort(areas)[::-1]
    props = [props[i] for i in srt]
    mask = np.zeros(well.shape)

    # add cells to mask using active contour
    for i in range(max_cells):
        if len(props) > 0:
            p = props[i].coords
            mask = np.zeros(b.shape)
            mask[p[:, 0], p[:, 1]] = 1
            c_tmp = find_contours(mask, 0, fully_connected='high', positive_orientation='high')[0]

            # artificially add more nodes for the active contours
            c = np.zeros((2 * c_tmp.shape[0], 2))
            c[::2] = c_tmp
            c[1::2] = c_tmp
            # find cell using active contour
            snake = active_contour(well, c, boundary_condition='periodic', alpha=alpha, beta=beta, gamma=gamma,
                                   max_num_iter=25)
            rows, cols = polygon(snake[:, 0], snake[:, 1], mask.shape)
            mask[rows, cols] = 1

    # find labels of regions
    labels, n_labs = label(mask, return_num=True, connectivity=1)
    props = [p for p in regionprops(labels) if p.area > 70 and p.eccentricity < .8]
    areas = [p.area for p in props]
    inds = np.argsort(areas)
    props = [props[i] for i in inds[-max_cells:]]

    # remove regions smaller than the top areas
    cell = np.zeros(well.shape, dtype=int)
    for p in props:
        cell[p.coords[:, 0], p.coords[:, 1]] = 1

    return cell, props


"""
    The code below defines which segmentation types will show up as options in the GUI itself
    Each entry for a segmentation type with have a name (the key in the main dictionary), which points to a list with 
    only two values:
        1. the function that will be called for segmentation
        2. a dictionary of keyword arguments that will be passed to the function; each key in this dictionary is a 
           keyword that needs to be passed to the function and the values are definitions needed in order to make a 
           Bokeh slider, containing:
                i. the label for the slider (what the user actually sees)
                ii. the lowest possible value on the slider
                iii. the highest possible value
                iv. the step size
                v. the starting position on the slider 
    
    This looks slightly clunky, but I've found that this is the easiest way to add new segmentation methods easily. For
    an example of the needed input/output of a segmentation method that can be used in this GUI, read the documentation
    of the "thresh_find_cell" function.
"""
segmentation_types = {
    'Thresholding': [thresh_find_cell, {'low_quant': ['High Threshold', 60., 100., .1, 92.],
                                        'high_quant': ['Low Threshold', 60., 100., .1, 97.],
                                        'erosion_rad': ['Erosion Amount', 1, 10, 1, 4]
                                        }
                     ],
    # 'Active Contours': [active_find_cell, {'quant': ['Threshold', 60., 100., .1, 80.],
    #                                        'alpha': ['Alpha', .01, 5., .01, .5],
    #                                        'beta': ['Beta', .01, 5., .01, 1.],
    #                                        'gamma': ['Gamma', 0.01, 5., .01, .01],
    #                                        'erosion_rad': ['Erosion Amount', 1, 10, 1, 4],
    #                                        }
    #                     ],
}


def find_all_cells(image: np.ndarray, cx, cy, radii, kwarg_params: dict, max_cells: int, type: str='Active Contours'):
    """
    Segments all of the wells in the given image
    :param image: the image containing the wells that need to be segmented (as a numpy ndarray)
    :param cx: the centers of the wells on the x-axis (as a list)
    :param cy: the centers of the wells on the y-axis (as a list)
    :param radii: the radiuses of the wells (as a list)
    :param kwarg_params: a dict of keyword arguments that should be passed on to the segmentation function
    :param max_cells: the maximum number of cells that can be found in each well (as an int)
    :param type: the type of segmentation function to use as a str - must be one of the keys in "segmentation_types"!
    :return: the tuple (wells, cells, props) where:
                1. wells: a list of images of all wells
                2. cells: a list of the segmentation of all of the cells in the wells (same order as wells)
                3. props: a list of the regionprops for each of the segmantations in "cells"
    """
    cells, wells, well_ints, props = [], [], [], []
    for i in range(len(cx)):
        # extract image of the well
        w = image[cy[i] - radii[i]:cy[i] + radii[i], cx[i] - radii[i]:cx[i] + radii[i]]
        # normalize between 0 and 1
        w = (w - np.min(w)) / (np.max(w) - np.min(w))
        # make an image of the circle where the edge of the well is
        c = circle((w.shape[0]//2, w.shape[1]//2), radii[0] - 30)

        well_int = np.quantile(w[c], q=.75) * np.ones(w.shape)
        well_int[c] = w[c]
        well_int = (well_int - np.min(well_int)) / (np.max(well_int) - np.min(well_int))
        well_int = well_int[:, :, 0]

        # calls segmentation function with the given parameters
        cell, ps = segmentation_types[type][0](w, c, **kwarg_params, max_cells=max_cells)
        cells.append(cell)
        wells.append(w)
        well_ints.append(well_int)
        props.append(ps)
    return wells, well_ints, cells, props


def find_circles(im: np.ndarray, radius: int, num_circles: int=35):
    """
    Find well locations in an image
    :param im: the image of the mold containing the wells
    :param radius: a rough estimate for the radius of the wells as an int (number of pixels)
    :param num_circles: number of wells expected in the image
    :return: the tuple (cx, cy, radii) where:
                1. cx: a list of the x-coordinates of the centers of the found wells
                2. cy: a list of the y-coordinates of the centers of the found wells
                3. radii: a list of the radii of the found wells
    """
    im = difference_of_gaussians(im, low_sigma=0, high_sigma=2)
    im = equalize_adapthist((im - np.min(im)) / (np.max(im) - np.min(im)), kernel_size=30)

    # get image edges
    edges = canny(im, sigma=.3, low_threshold=.4, high_threshold=.7, use_quantiles=True)

    # find wells
    hough_radii = [radius]
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=num_circles,
                                               min_xdistance=int(radius), min_ydistance=int(radius),
                                               normalize=True)

    cx, cy, radii = np.array(cx), np.array(cy), np.array(radii)
    inds = np.argsort(cy)
    return cx[inds], cy[inds], radii[inds]


def natural_order(corner: tuple, cx: np.ndarray, cy: np.ndarray):
    """
    Function that defines the ordering of the wells
    :param corner: well to use as corner
    :param cx: well centers on the x-axis
    :param cy: well centers on the y-axis
    :return: a natural ordering for numbering of the wells for
    """
    corner = np.array(corner)
    centers = np.concatenate([cx[:, None], cy[:, None]], axis=-1)
    # diag = np.sqrt(np.max(np.sum((centers - corner[None, :])**2, axis=-1)))
    #
    # centers = mold_diag*centers/diag
    # pnt = centers[np.argmax(np.sum(centers**2, axis=1))]
    # thet = np.arctan(7/5) - np.arctan(pnt[1]/pnt[0])
    # if pnt[1] > corner[1]: thet = -thet
    # centers = np.round(centers @ np.array([[np.cos(thet), -np.sin(thet)], [np.sin(thet), np.cos(thet)]]).T, 2)
    # return np.lexsort((centers[:, 0], centers[:, 1]))
    return np.argsort(np.sum((centers - corner[None, :])**2, axis=-1))


def well_intensity(well_int: np.ndarray):
    thresh = 1-well_int[0, 0]
    well_int = 1 - (well_int - np.min(well_int))/(np.max(well_int) - np.min(well_int))
    intensity = np.sum(well_int[well_int != thresh])
    return intensity
