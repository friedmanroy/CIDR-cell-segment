import numpy as np
import pickle
from pathlib import Path
from bokeh.plotting import figure, show, curdoc
from bokeh.events import Tap
from bokeh.models import Slider, Button, Spinner, MultiSelect, Select
from bokeh.layouts import column, layout, row
from bokeh.models.tools import WheelZoomTool, PointDrawTool, PolyDrawTool, PolyEditTool
from bokeh.server.server import Server
from bokeh.settings import settings
from skimage.segmentation import mark_boundaries
from skimage.measure import find_contours, approximate_polygon
from skimage.draw import polygon2mask
from skimage import color
from skimage.transform import rescale
from aux_funcs import (find_all_cells, find_circles, rgb2rgba, read_im, natural_order, segmentation_types,
                       well_intensity)
from tkinter import messagebox, filedialog, Tk, simpledialog
from matplotlib import pyplot as plt, patches
import os, csv, functools, logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def log(func):
    """
    Logger decoration function for debugging the Bokeh GUI
    :param func: the function to wrap with the logger
    :return: the function wrapped with the logger
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # when debugging, add function name and all input and keyword variables
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.debug(f"function {func.__name__} called with args {signature}")

        # run the function
        try:
            result = func(*args, **kwargs)
            return result
        # Bokeh doesn't always stop on exceptions, so this is a way to see where an exception was raised
        except Exception as e:
            logger.exception(f"Exception raised in {func.__name__}. exception: {str(e)}")
            raise e
    return wrapper


def segment_sliders(callback):
    """
    Generate all the segmentation algorithms' sliders and buttons
    :param callback: the function to call when sliders are changed
    :return: a dictionary of sliders and another dictionary of the keys to relevant information
    """
    buttons = dict()
    keys = dict()

    # for each segmentation algorithm, add the control buttons
    for seg_type in segmentation_types:
        btns = []

        # the following dictionary contains all of the needed sliders
        slider_dict = segmentation_types[seg_type][1]
        for k in slider_dict:
            lst = slider_dict[k]
            sld = Slider(title=lst[0], start=lst[1], end=lst[2], step=lst[3], value=lst[4],
                         sizing_mode='scale_width')
            sld.on_change('value_throttled', lambda x, y, z: callback())
            btns.append(sld)
        buttons[seg_type] = column(*btns)
        keys[seg_type] = list(segmentation_types[seg_type][1].keys())
    return buttons, keys


def _empty_event(event=None):
    """
    Helper function (used mainly for debugging) of an empty Bokeh event
    """
    pass


def _get_radius(n_wells: int):
    if n_wells == 7: pix_rad, hough_sc = 18, .2
    elif n_wells == 21: pix_rad, hough_sc = 11, .2
    elif n_wells == 25: pix_rad, hough_sc = 7, .2
    else: pix_rad, hough_sc = 30, .2
    return pix_rad


# constants used in GUI
SEGMENTATION_TYPES = list(segmentation_types.keys())
WELLS, AUTOMATIC, FIXES = 0, 1, 2
POLY_DRAW_TEXT = 'Draw or move ROIs'
POLY_EDIT_TEXT = 'Change ROI shapes'

# set to 'trace' in order to follow all possible leads for problems
settings.log_level('trace')
settings.py_log_level('trace')


class SimpleGUI:
    standard_len = 7000
    standard_nwells = 35
    well_rad = 400
    button_size = 400
    ext_list = ['jpg']

    @log
    def __init__(self, im_path: str, scale: float=None, n_wells: int=35):
        """
        Initialize the GUI using the given image path and scale
        :param im_path: the path to the image to open in the GUI
        :param scale: the scale of the image to be shown
        :param n_wells: number of wells in the mold
        """
        self.im_path = im_path
        # scale images to the same size (1500 pixels on long end)
        if scale is None: scale = 1500/np.max(plt.imread(im_path).shape)
        self.scale = scale

        # read image and find wells
        self.im = read_im(im_path, scale)
        self._find_circles(n_wells, _get_radius(n_wells))
        self.mask = None

        # define default lengths
        self.calc_len = 1024.13 * self.scale / .4
        self.pix2micron = self.standard_len / self.calc_len

        # helper attributes
        self.page = WELLS
        self._tap_event = _empty_event
        self.fixing_wells = False
        self.drawing_new = False
        self.drawn_segments = []
        self.segment_dict, self.segment_keys = segment_sliders(self.refresh_GUI)

        # build layout
        self._build_layout()
        self.update_wells()

    @log
    def _find_circles(self, n_circles: int, well_rad: int=27):
        """
        Find the well placement and update in the GUI
        """
        scale = .2

        # find wells
        cx, cy, radii = find_circles(rescale(color.rgb2gray(self.im.copy()), scale/self.scale),
                                     well_rad, num_circles=n_circles)

        # rescale centers and radii according to scale
        self.cx = (self.scale*cx/scale).astype(int)
        self.cy = (self.scale*cy/scale).astype(int)
        self.radii = (self.scale*radii/scale).astype(int)
        # choose initial corner
        self.corner = np.argmin(self.cy)
        # reorder wells
        ord = natural_order((self.cx[self.corner], self.cy[self.corner]), self.cx, self.cy)
        self.cx, self.cy, self.radii = self.cx[ord], self.cy[ord], self.radii[ord]

    @log
    def _build_layout(self):
        """
        Builds the layout of the whole GUI
        """
        self._build_buttons()
        self._init_plot()
        self._init_small_plot()
        self.layout = layout(column(
            [row([self.figure,
                 column(self.buttons_handles,
                        self.small_figure,
                        self.well_slider),
                 ], height_policy='max'),
             ]))
        self.refresh_GUI()

    @log
    def _init_plot(self):
        """
        Initializes the (big) image, which will then allow seamless refreshing
        """
        self.figure = figure(tools='pan, reset', min_border=0,
                             toolbar_location='right', x_axis_location=None, y_axis_location=None,
                             sizing_mode='scale_height')
        self.figure.toolbar.logo = None

        self.figure.x_range.bounds = [0, self.im.shape[1]]
        self.figure.y_range.bounds = [0, self.im.shape[0]]
        self.figure.x_range.range_padding = self.figure.y_range.range_padding = 0

        def callback(event): self._tap_event(event)
        self.figure.on_event(Tap, callback)

        # add image
        self.im_glyph = self.figure.image_rgba('image', image=[rgb2rgba(self.im).astype(np.uint8)], x=0, y=0,
                                               dw=self.im.shape[1], dh=self.im.shape[0], dilate=False)
        # add circles with their text
        colors = ['red' if i!=self.corner else 'blue' for i in range(len(self.cx))]
        self.circles = self.figure.circle(self.cx, self.im.shape[1]-self.cy, radius=self.radii,
                                          fill_alpha=0, line_color=colors, line_width=2.5)
        self.text = self.figure.text(x=self.cx-self.radii+5, y=self.im.shape[1]-(self.cy-self.radii+5),
                                     text=np.arange(len(self.cx))+1, text_color='red')

        # add special tools
        wheel_zoom = WheelZoomTool(maintain_focus=False)
        self.figure.add_tools(wheel_zoom)
        self.figure.toolbar.active_scroll = wheel_zoom

        self.polygons = self.figure.patches(xs=[[]], ys=[[]], line_color='red', line_width=2,
                                            fill_alpha=0.15, fill_color='red')
        self.polydraw_tool = PolyDrawTool(renderers=[self.polygons])
        self.figure.add_tools(self.polydraw_tool)
        self.figure.tools = self.figure.tools[:-1]

        self.vert_rend = self.figure.circle([], [], size=15, color='red')
        self.polyedit_tool = PolyEditTool(renderers=[self.polygons], vertex_renderer=self.vert_rend)
        self.figure.add_tools(self.polyedit_tool)
        self.figure.tools = self.figure.tools[:-1]

    @log
    def _init_small_plot(self):
        """
        Initializes the zoom-in images of the wells
        """
        self.small_figure = figure(min_border=0, tools='', x_axis_location=None,
                                   y_axis_location=None, plot_width=self.button_size,
                                   plot_height=self.button_size)
        well = self.im[self.cx[0] - self.radii[0]:self.cx[0] + self.radii[0],
                       self.cy[0] - self.radii[0]:self.cy[0] + self.radii[0]]
        self.small_figure.x_range.range_padding = self.small_figure.y_range.range_padding = 0
        self.small_figure.toolbar.logo = None
        self.small_figure.toolbar_location = None
        self.small_figure.x_range.bounds = [0, well.shape[1]]
        self.small_figure.y_range.bounds = [0, well.shape[0]]
        self.small_figure.image_rgba('image', image=[rgb2rgba(well).astype(np.uint8)], x=0, y=0, dw=well.shape[1],
                                     dh=well.shape[0], dilate=True)

        # add a slider to move across all wells
        self.well_slider = Slider(title='', start=1, end=len(self.cx), step=1, value=1,
                                  sizing_mode='scale_width')
        self.well_slider.on_change('value', lambda x, y, z: self._draw_well())

    @log
    def _build_buttons(self):
        """
        Builds all of the buttons in the GUI, in the 3 different pages
        """
        self._pages = {}

        # ==== save button
        self.save_button = Button(label='Save', default_size=150, sizing_mode='scale_width')
        self.save_button.on_click(self.save_event)
        self.save_button.disabled = True

        # ==== open image button
        self.open_button = Button(label='Open new file', default_size=150, sizing_mode='scale_width')
        self.open_button.on_click(self.open_event)

        save_row = row([self.save_button, self.open_button])

        # ==== next page button
        self.next_button = Button(label='Next step', default_size=150, sizing_mode='scale_width')
        self.next_button.on_click(self.next_event)

        # ==== restart process button
        self.restart_button = Button(label='Restart process', default_size=150, sizing_mode='scale_width')
        self.restart_button.on_click(lambda: self.next_event(restart=True))

        next_row = row([self.next_button, self.restart_button])

        # ------------------------------------------------------------------------------- create first page
        # ==== mold corner definition button

        self.num_wells_spinn = Spinner(title='Number of wells:', low=1, high=50, step=1, value=35,
                                       sizing_mode='scale_width')
        self.num_wells_spinn.on_change('value', lambda x, y, z: self._rad_setter())

        self.well_rad_spinn = Spinner(title='Well radius:', low=1, high=100, step=1,
                                      value=_get_radius(self.standard_nwells), sizing_mode='scale_width')
        self.well_rad_spinn.on_change('value', lambda x, y, z: _empty_event())

        self.find_button = Button(label='Find wells', default_size=150, sizing_mode='scale_width')
        self.find_button.on_click(self.update_wells)

        self.corner_button = Button(label='Mark corner', default_size=150, sizing_mode='scale_width')
        self.corner_button.on_click(self.choose_corner_event)

        # ==== button to start fixing circle placements
        self.circ_fix = Button(label='Fix well locations', default_size=150, sizing_mode='scale_width')
        self.circ_fix.on_click(self.fix_well_event)

        # create measurement page
        self.meas_row = row([self.corner_button, self.circ_fix, self.find_button])
        self.well_row = row([self.num_wells_spinn, self.well_rad_spinn])
        self._pages[WELLS] = [self.meas_row, self.well_row, next_row, save_row]

        # ------------------------------------------------------------------------------- create second page
        # define buttons used in all schemes
        self.segmentation_select = Select(title='Segmentation Method:', value=SEGMENTATION_TYPES[0],
                                          options=SEGMENTATION_TYPES)
        self.segmentation_select.on_change('value', lambda x, y, z: self.change_segmentation_alg_event())

        self.cell_spinner = Spinner(title='Max number of aggregates:', low=1, high=10, step=1, value=1,
                                    sizing_mode='scale_width')
        self.cell_spinner.on_change('value', lambda x, y, z: self.refresh_GUI())

        aut_buttons = [self.segmentation_select,
                       self.segment_dict[self.segmentation_select.value],
                       self.circ_fix,
                       next_row,
                       save_row]
        self._pages[AUTOMATIC] = [*aut_buttons]

        # ------------------------------------------------------------------------------- create final page
        self.draw_button = Button(label=POLY_DRAW_TEXT, default_size=150, sizing_mode='scale_width')
        self.draw_button.on_click(self.draw_event)

        self.edit_button = Button(label=POLY_EDIT_TEXT, default_size=150, sizing_mode='scale_width')
        self.edit_button.on_click(lambda: self.draw_event(edit=True))

        self.seg_row = row([self.edit_button, self.draw_button])

        self._pages[FIXES] = [self.seg_row, self.restart_button, save_row]

        # ------------------------------------------------------------------------------- put everything together
        # define current buttons and all buttons
        self.buttons_handles = column(self._pages[self.page], width=self.button_size)
        self.buttons = [
            self.save_button,
            self.next_button,
            self.draw_button,
            self.corner_button,
            self.cell_spinner,
            *[c for k in self.segment_dict for c in self.segment_dict[k].children],
            self.circ_fix,
            self.restart_button,
            self.open_button,
            self.edit_button
        ]

    @log
    def _find_well(self, x, y):
        """
        Helper function to find which well is currently selected
        :param x: x position of the mouse
        :param y: y position of the mouse
        :return: the well number which is closest to the event
        """
        return np.argmin((self.cx-x)**2 + (self.cy-y)**2)

    @log
    def _draw_well(self):
        """
        Plots the zoom-in version of the current well with the (current) segmentation in red
        """
        # controls amount of padding in the zoom-in images
        bd = 0

        # finds currently selected well
        ind = self.well_slider.value - 1

        # slices the image for the current well and corresponding segmentation mask
        well = self.im[self.cy[ind] - self.radii[ind] + bd:self.cy[ind] + self.radii[ind] - bd,
                       self.cx[ind] - self.radii[ind] + bd:self.cx[ind] + self.radii[ind] - bd]
        mask = self.mask[self.cy[ind] - self.radii[ind] + bd:self.cy[ind] + self.radii[ind] - bd,
                         self.cx[ind] - self.radii[ind] + bd:self.cx[ind] + self.radii[ind] - bd]
        # mark the segmentation of the image
        well_marked = mark_boundaries(well, mask, color=(1, 0, 0), mode='thick')
        # controls how strong the segmentation will look on the zoom-in
        alpha = .5
        if self.page != WELLS:
            self.small_figure.renderers[0].data_source.data.update({
                'image': [rgb2rgba((1-alpha)*well + alpha*well_marked).astype(np.uint8)]
            })
        else:
            self.small_figure.renderers[0].data_source.data.update({'image': [rgb2rgba(well).astype(np.uint8)]})

    @log
    def _convert_to_poly(self):
        """
        Converts the segmentations from masks into editable polygons
        """
        poly_xs = []
        poly_ys = []
        for i in range(len(self.cells)):
            if np.any(self.cells[i]):
                # find contour lines in mask
                conts = find_contours(self.cells[i], 0, fully_connected='high', positive_orientation='high')
                for cont in conts:
                    # approximate contour with polygon
                    co = approximate_polygon(cont, tolerance=1)
                    poly_xs.append(co[:, 1] + self.cx[i] - self.radii[i])
                    poly_ys.append(self.im.shape[1] - (co[:, 0] + self.cy[i] - self.radii[i]) + 3)
        # add all polygons to the GUI
        self.polygons.data_source.data.update({'xs': poly_xs, 'ys': poly_ys})

    @log
    def _convert_to_mask(self):
        """
        Converts editable polygons back into masks for area calculation and saving
        :return: wells - a list of the well numbers corresponding to the areas
                 areas - a list of the areas of the cells, according to the segmentation
                 mask - a np.ndarray, with the same shape as the full image, which is a mask of all segmentations
                        together
        """
        mask = np.zeros(self.mask.shape)
        wells, areas = [], []
        for i in range(len(self.polygons.data_source.data['xs'])):
            x = np.array(self.polygons.data_source.data['xs'][i]).squeeze()
            y = np.array(self.polygons.data_source.data['ys'][i]).squeeze()
            wells.append(np.argmin((np.mean(x)-self.cx)**2 + (self.im.shape[1] - np.mean(y)-self.cy)**2) + 1)
            tmp = polygon2mask(mask.shape, np.concatenate([self.im.shape[1] - y[:, None],
                                                           x[:, None]],
                                                          axis=1))
            areas.append(np.sum(tmp))
            mask += tmp
        return wells, areas, mask

    @log
    def _find_conversion_ratio(self):
        # below is the old way of calculating the micron to pixel ratio, which was mold dependent (keeping just in case)
        # pt = np.array([self.cx[self.corner], self.cy[self.corner]])
        # dists = np.sqrt(np.sum((pt[:, None] - np.stack([self.cx, self.cy]))**2, axis=0))
        # self.calc_len = (np.max(dists) + 2*self.radii[self.corner])
        # self.pix2micron = self.standard_len / self.calc_len

        # wells have a radius of 400 microns, so we use that in order to calculate the micron to pixel ratio
        self.pix2micron = self.well_rad / self.radii[self.corner]

    @log
    def _save_to_file(self, file):
        """
        Saves all of the segmentation data into files
        :param file: the file name (excluding extensions) used to save
        """
        wells, areas, mask = self._convert_to_mask()
        self._find_conversion_ratio()
        # write areas to a .csv file
        with open(file + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['well',
                             'area (px^2)',
                             'area (micron^2)',
                             'well intensity',
                             'well x center (px)',
                             'well y center (px)'])
            for i in range(len(wells)):
                writer.writerow([wells[i],
                                 areas[i],
                                 np.round(areas[i]*(self.pix2micron**2)/(self.scale**2), 2),
                                 np.round(well_intensity(self.well_ints[i]), 2),
                                 self.cx[wells[i]-1],
                                 self.cy[wells[i]-1]])
            writer.writerow(['Misc:', 'Calc. length (px):', self.calc_len])
            writer.writerow(['', 'px/micron ratio:', self.pix2micron])

        # create image with segmentation labelings (numbers and segmentation boundary)
        fig = plt.figure(dpi=300)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        image = mark_boundaries(self.im, mask, color=(1, 0, 0), mode='thick')
        plt.imshow(image)

        data = {'masks': [], 'images': []}
        for i, (cy, cx, rad) in enumerate(zip(self.cy, self.cx, self.radii)):
            data['masks'].append(mask[cy - rad:cy + rad, cx - rad:cx + rad])
            data['images'].append(color.rgb2gray(self.im[cy - rad:cy + rad, cx - rad:cx + rad]))
            circ = patches.Circle((cx, cy), rad, facecolor='none', edgecolor='r', lw=.5)
            ax.add_patch(circ)
            ax.text(cx - rad, cy - rad, str(i + 1), fontsize=4, color='r')
        plt.savefig(file + '.jpg')

        # save data needed in order to load back all data (not used at the moment)
        pkl_path = str(Path(file).parent) + '/.__raw_segmented_data/'
        Path(pkl_path).mkdir(exist_ok=True)
        with open(pkl_path + Path(file).name + '.pkl', 'wb') as f: pickle.dump(data, f)

    @log
    def _rad_setter(self):
        val = _get_radius(self.num_wells_spinn.value)
        self.well_rad_spinn.value = val

    @log
    def disable_buttons(self):
        for button in self.buttons: button.disabled = True

    @log
    def activate_buttons(self):
        for button in self.buttons: button.disabled = False
        self.save_button.disabled = True
        if self.page == FIXES: self.save_button.disabled = False

    @log
    def fix_well_event(self):
        """
        Event for when the "Fix well locations" button is pressed
        """
        # if the button was previously pressed, return everything back to the normal state
        if self.fixing_wells:
            # remove drag tool from toolbar
            self.figure.tools = self.figure.tools[:-1]

            # extract the circle's new locations
            dat = self.circles.data_source.data
            self.cx = dat['x']
            self.cy = self.im.shape[1] - dat['y']

            # update text locations
            self.text.data_source.data['x'] = self.cx - self.radii + 5
            self.text.data_source.data['y'] = self.im.shape[1] - (self.cy - self.radii + 5)

            # switch back to original button and draw
            ord = natural_order((self.cx[self.corner], self.cy[self.corner]), self.cx, self.cy)
            self.cx, self.cy, self.radii = self.cx[ord], self.cy[ord], self.radii[ord]
            self.fixing_wells = False
            self.circ_fix.label = 'Fix well locations'

            self.refresh_GUI()

        # if the button is unpressed, move to fixing locations mode
        else:
            # change flag and block all buttons
            self.fixing_wells = True
            self.disable_buttons()
            self.circ_fix.disabled = False
            self.circ_fix.label = 'Stop'

            # make circles draggable
            draw_tool = PointDrawTool(add=False, renderers=[self.circles])
            self.figure.add_tools(draw_tool)
            self.figure.toolbar.active_tap = draw_tool
            self._draw_well()

    @log
    def change_segmentation_alg_event(self):
        aut_buttons = [self.segmentation_select,
                       self.segment_dict[self.segmentation_select.value],
                       self.circ_fix,
                       row([self.next_button, self.restart_button]),
                       row([self.save_button, self.open_button])]
        self._pages[AUTOMATIC] = [*aut_buttons]

        # update buttons to the correct set
        self.buttons_handles.children = self._pages[AUTOMATIC]
        self.refresh_GUI()

    @log
    def save_event(self):
        # create yes-no message box if everything should be removed (in focus)
        root = Tk()
        root.deiconify()
        root.lift()
        root.focus_force()
        f = filedialog.asksaveasfile()
        root.destroy()

        if f and len(str(f)) > 0:
            self._save_to_file(f.name)

    @log
    def update_wells(self):
        self.circles.visible = False
        self.text.visible = False
        self._find_circles(self.num_wells_spinn.value, well_rad=self.well_rad_spinn.value)

        colors = ['red' if i != self.corner else 'blue' for i in range(len(self.cx))]
        self.circles = self.figure.circle(self.cx, self.im.shape[1] - self.cy, radius=self.radii,
                                          fill_alpha=0, line_color=colors, line_width=2.5)
        self.text = self.figure.text(x=self.cx - self.radii + 5, y=self.im.shape[1] - (self.cy - self.radii + 5),
                                     text=np.arange(len(self.cx)) + 1, text_color='red')

    @log
    def open_event(self):
        root = Tk()
        root.deiconify()
        root.lift()
        root.focus_force()
        path = filedialog.askopenfilename(filetypes=[('images', '.jpg .tif'), ('JPG', '*.jpg')])
        root.destroy()

        root = Tk()
        root.deiconify()
        root.lift()
        root.focus_force()
        n_wells = simpledialog.askinteger('Number of Wells', 'Number of wells to use:', parent=root, initialvalue=35)
        root.destroy()

        if path != '':
            try:
                self.disable_buttons()
                # self.circles.visible = False
                # self.text.visible = False

                self.im = read_im(path, self.scale, match_size=self.im.shape)
                # self._find_circles(n_wells, well_rad=self.well_rad_spinn)
                self.mask = None

                # add new image to plot
                self.figure.renderers[0].data_source.data.update({'image': [rgb2rgba(self.im)]})
                self.im_glyph.glyph.update(x=0, y=0, dw=self.im.shape[1], dh=self.im.shape[0])

                self.figure.x_range.bounds = [0, self.im.shape[1]]
                self.figure.y_range.bounds = [0, self.im.shape[0]]
                self.figure.x_range.range_padding = self.figure.y_range.range_padding = 0

                # # add circles with their text
                # colors = ['red' if i != self.corner else 'blue' for i in range(len(self.cx))]
                # self.circles = self.figure.circle(self.cx, self.im.shape[1] - self.cy, radius=self.radii,
                #                                   fill_alpha=0, line_color=colors, line_width=2.5)
                # self.text = self.figure.text(x=self.cx - self.radii + 5, y=self.im.shape[1]-(self.cy - self.radii + 5),
                #                              text=np.arange(len(self.cx)) + 1, text_color='red')

                self.next_event(restart=True, ask=False)
                self.activate_buttons()

            except FileNotFoundError:
                root = Tk()
                root.deiconify()
                root.lift()
                root.focus_force()
                answer = messagebox.askyesno('File not found!', 'The file you tried to open was not found,'
                                                                'open new image?', parent=root)
                root.destroy()
                if answer: self.open_event()

    @log
    def next_event(self, restart: bool=False, ask: bool=True):
        if self.page == WELLS:
            self.page = AUTOMATIC
            self.save_button.disabled = True
        elif self.page == AUTOMATIC:
            self.save_button.disabled = False
            self.page = FIXES
            self._convert_to_poly()

        # if process is restarted, remove everything
        if restart:
            if ask:
                # create yes-no message box if everything should be removed (in focus)
                root = Tk()
                root.deiconify()
                root.lift()
                root.focus_force()
                answer = messagebox.askyesno('Question', 'All unsaved changed will be lost. '
                                                         'Are you sure you want to continue?', parent=root)
                root.destroy()
            else: answer = True
            if answer:
                self.page = WELLS
                self.polygons.data_source.data = {'xs': [], 'ys': []}
                self.save_button.disabled = True

        # update buttons to the correct set
        self.buttons_handles.children = self._pages[self.page]
        self.refresh_GUI()

    @log
    def draw_event(self, edit: bool=False):
        if not self.drawing_new:
            # switch draw button label to 'Stop' and disable all buttons
            self.drawing_new = not self.drawing_new
            self.disable_buttons()
            button = self.edit_button if edit else self.draw_button
            button.label = 'Stop'
            button.disabled = False

            # activate the PolyDrawTool if not editing, otherwise the PolyEditTool
            self.figure.add_tools(self.polyedit_tool if edit else self.polydraw_tool)
            if edit: self.figure.toolbar.active_drag = self.polyedit_tool
            else: self.figure.toolbar.active_tap = self.polydraw_tool
        else:
            # remove PolyDrawTool from toolbar
            self.figure.toolbar.active_tap = None
            self.figure.toolbar.active_drag = self.figure.tools[0]
            self.figure.tools = self.figure.tools[:-1]

            # remove vertex renders
            data = {i: self.vert_rend.data_source.data[i] for i in self.vert_rend.data_source.data.keys()}
            data['x'], data['y'] = [], []
            self.vert_rend.data_source.data = data

            # return button to original state and activate buttons
            self.drawing_new = not self.drawing_new
            if edit: self.edit_button.label = POLY_EDIT_TEXT
            else: self.draw_button.label = POLY_DRAW_TEXT
            self.activate_buttons()

    @log
    def refresh_GUI(self):
        self.disable_buttons()
        seg_type = self.segmentation_select.value
        segment_args = self.segment_dict[seg_type]
        segment_args = {self.segment_keys[seg_type][i]: c.value for i, c in enumerate(segment_args.children)}
        self.wells, self.well_ints, self.cells, self.props = find_all_cells(self.im.copy(),
                                                                            self.cx.copy(),
                                                                            self.cy.copy(),
                                                                            self.radii.copy(),
                                                                            segment_args,
                                                                            max_cells=self.cell_spinner.value,
                                                                            type=self.segmentation_select.value)

        self.mask = np.zeros(self.im.shape[:-1]).astype(int)
        for i, (cy, cx, r) in enumerate(zip(self.cy, self.cx, self.radii)):
            self.mask[cy - r:cy + r, cx - r:cx + r] = self.cells[i]

        image = mark_boundaries(self.im, self.mask, color=(1,0,0), mode='thick') if self.page == AUTOMATIC else self.im

        self.figure.renderers[0].data_source.data.update({'image': [(rgb2rgba(image)).astype(np.uint8)]})
        self._draw_well()

        self.activate_buttons()

    @log
    def choose_corner_event(self):
        self.disable_buttons()

        def callback(event):
            # make all circles red
            d = self.circles.data_source.data
            d['line_color'] = ['red']*len(d['line_color'])

            # find the new corner
            dists = (self.cx-event.x)**2 + (self.cy-self.im.shape[1]+event.y)**2
            c1, c2 = self.cx[np.argmin(dists)], self.cy[np.argmin(dists)]

            # reorganize according to the corner placement
            ord = natural_order((c1, c2), self.cx, self.cy)
            # ord = np.argsort(np.abs(self.cx-c1) + np.abs(self.cy-c2))
            self.cx, self.cy, self.radii = self.cx[ord], self.cy[ord], self.radii[ord]
            d['x'], d['y'] = self.cx, self.im.shape[1]-self.cy
            d['line_color'][0] = 'blue'
            self.circles.data_source.data = {a: d[a] for a in d.keys()}

            # rewrite text according to order
            text_data = {'x': self.cx-self.radii+5,
                         'y': self.im.shape[1]-(self.cy-self.radii+5),
                         'text': np.arange(len(self.cx))+1}
            self.text.data_source.data = text_data

            self._draw_well()
            self.activate_buttons()
            self._tap_event = _empty_event
        self._tap_event = callback

    @log
    def serve(self, doc):
        doc.add_root(self.layout)

    @staticmethod
    def _get_dir_files(directory: str, ext_list: list):
        return ['⏎'] + [f if not os.path.isdir(f) else f'{f}/' for f in os.listdir(directory)
                        if (os.path.isdir(f) or f.split('.')[-1] in ext_list)
                        and (not f.startswith('.')) and (not f.startswith('__'))]


# root = Tk()
# root.deiconify()
# root.lift()
# root.focus_force()
# path = filedialog.askopenfilename(filetypes=[('JPG', '*.jpg')])
# root.destroy()
# f = SimpleGUI(im_path=path)
# curdoc().add_root(f.layout)
# curdoc().on_session_destroyed(lambda _: sys.exit())

if __name__ == '__main__':
    root = Tk()
    root.deiconify()
    root.lift()
    root.focus_force()
    path = filedialog.askopenfilename(filetypes=[('images', '.jpg .tif'), ('JPG', '*.jpg'), ('TIF', '*.tif')])
    root.destroy()

    if path != '':
        # root = Tk()
        # root.deiconify()
        # root.lift()
        # root.focus_force()
        # n_vals = simpledialog.askinteger('Number of Wells', 'Number of wells to use:', parent=root, initialvalue=35)
        # root.destroy()

        # if n_vals is not None:
        f = SimpleGUI(im_path=path)
        server = Server({'/': f.serve})
        server.start()

        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()
        server.run_until_shutdown()
