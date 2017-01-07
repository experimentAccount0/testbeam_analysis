from __future__ import division

import logging
import re
import os.path
import warnings
from math import ceil

import numpy as np
import tables as tb
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib import colors, cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3d plotting although it is shown as not used
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider, Button
from scipy.optimize import curve_fit

import testbeam_analysis.tools.analysis_utils

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")  # Plot backend error not important


def plot_2d_pixel_hist(fig, ax, hist2d, plot_range, title=None, x_axis_title=None, y_axis_title=None, z_min=0, z_max=None):
    extent = [0.5, plot_range[0] + .5, plot_range[1] + .5, 0.5]
    if z_max is None:
        if hist2d.all() is np.ma.masked:  # check if masked array is fully masked
            z_max = 1
        else:
            z_max = ceil(hist2d.max())
    bounds = np.linspace(start=z_min, stop=z_max, num=255, endpoint=True)
    cmap = cm.get_cmap('viridis')
    cmap.set_bad('w')
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(hist2d, interpolation='none', aspect="auto", cmap=cmap, norm=norm, extent=extent)
    if title is not None:
        ax.set_title(title)
    if x_axis_title is not None:
        ax.set_xlabel(x_axis_title)
    if y_axis_title is not None:
        ax.set_ylabel(y_axis_title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, boundaries=bounds, cmap=cmap, norm=norm, ticks=np.linspace(start=z_min, stop=z_max, num=9, endpoint=True), cax=cax)


def plot_noisy_pixels(input_hits_file, pixel_size=None, output_pdf_file=None, dut_name=None):
    with tb.open_file(input_hits_file, 'r') as input_file_h5:
        occupancy = np.ma.masked_array(input_file_h5.root.HistOcc[:], mask=input_file_h5.root.NoisyPixelsMask[:])

    if pixel_size:
        aspect = pixel_size[0] / pixel_size[1]
    else:
        aspect = "auto"

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_hits_file)[0] + '.pdf'

    if not dut_name:
        dut_name = os.path.split(input_hits_file)[1]

    with PdfPages(output_pdf_file) as output_pdf:
        plt.figure()
        ax = plt.subplot(111)

        cmap = cm.get_cmap('viridis')
    #     cmap.set_bad('w')
    #     norm = colors.LogNorm()
        norm = None
        c_max = np.percentile(occupancy, 99)

        noisy_pixels = np.nonzero(np.ma.getmaskarray(occupancy))
        # check for any noisy pixels
        if noisy_pixels[0].shape[0] != 0:
            ax.plot(noisy_pixels[1], noisy_pixels[0], 'ro', mfc='none', mec='r', ms=10)
        ax.set_title('%s with %d noisy pixel' % (dut_name, np.ma.count_masked(occupancy)))
        ax.imshow(np.ma.getdata(occupancy), aspect=aspect, cmap=cmap, norm=norm, interpolation='none', origin='lower', clim=(0, c_max))
        ax.set_xlim(-0.5, occupancy.shape[1] - 0.5)
        ax.set_ylim(-0.5, occupancy.shape[0] - 0.5)

        output_pdf.savefig()

        plt.figure()
        ax = plt.subplot(111)

        ax.set_title('%s with %d noisy pixel removed' % (dut_name, np.ma.count_masked(occupancy)))
        ax.imshow(occupancy, aspect=aspect, cmap=cmap, norm=norm, interpolation='none', origin='lower', clim=(0, c_max))
    #     np.ma.filled(occupancy, fill_value=0)
        ax.set_xlim(-0.5, occupancy.shape[1] - 0.5)
        ax.set_ylim(-0.5, occupancy.shape[0] - 0.5)

        output_pdf.savefig()
        plt.close()


def plot_cluster_size(input_cluster_file, output_pdf_file=None, dut_name=None):
    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_cluster_file)[0] + '.pdf'

    if not dut_name:
        dut_name = os.path.split(input_cluster_file)[1]

    with PdfPages(output_pdf_file) as output_pdf:
        with tb.open_file(input_cluster_file, 'r') as input_file_h5:
            cluster_n_hits = input_file_h5.root.Cluster[:1000000]['n_hits']
            # Save cluster size histogram
            max_cluster_size = np.amax(cluster_n_hits) + 1
            plt.clf()
            left = np.arange(max_cluster_size)
            hight = testbeam_analysis.tools.analysis_utils.hist_1d_index(cluster_n_hits, shape=(max_cluster_size,))
            plt.bar(left, hight, align='center')
            plt.title('Cluster size of %s' % dut_name)
            plt.xlabel('Cluster size')
            plt.ylabel('#')
            plt.grid()
            plt.yscale('log')
            plt.xlim(xmin=0.5)
            plt.ylim(ymin=1e-1)
            output_pdf.savefig()
            plt.yscale('linear')
            plt.ylim(ymax=np.amax(hight))
            plt.xlim(0.5, min(10, max_cluster_size - 1) + 0.5)
            output_pdf.savefig()


def plot_correlation_fit(x, y, y_fit, xlabel, fit_label, title, output_pdf):
    plt.clf()
    plt.plot(x, y, 'r.-', label='Data')
    plt.plot(x, y_fit, 'g-', linewidth=2, label='Fit: %s' % fit_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('#')
    plt.xlim((np.min(x), np.max(x)))
    plt.grid()
    plt.legend(loc=0)
    if output_pdf:
        output_pdf.savefig()
    else:
        plt.show()


def plot_coarse_alignment_check(column_0, column_1, row_0, row_1, corr_x, corr_y, dut_index, output_pdf):
    # Plot column check
    plt.clf()
    plt.title('Coarse alignment check, column of DUT%d against reference DUT' % dut_index)
    median = np.median(column_1 - column_0)
    std = np.std(column_1 - column_0)
    y, _, _ = plt.hist(column_1 - column_0, bins=100, range=(-2 * std, 2 * std), label='Data')
    plt.plot([median, median], [0, y.max()], label='Median %1.2f um' % median)
    plt.plot([median - corr_x, median - corr_x], [0, y.max()], '--', color='red', label='68 % of the data')
    plt.plot([median + corr_x, median + corr_x], [0, y.max()], '--', color='red')
    plt.grid()
    plt.xlabel('Difference [um]')
    plt.ylabel('#')
    plt.yscale('log')
    plt.legend(loc=0)
    output_pdf.savefig()
    # Plot row check
    plt.clf()
    plt.title('Coarse alignment check, row of DUT%d against reference DUT' % dut_index)
    median = np.median(row_1 - row_0)
    std = np.std(row_1 - row_0)
    y, _, _ = plt.hist(row_1 - row_0, bins=100, range=(-2 * std, 2 * std), label='Data')
    plt.plot([median, median], [0, y.max()], label='Median %1.2f um' % median)
    plt.plot([median - corr_x, median - corr_x], [0, y.max()], '--', color='red', label='68 % of the data')
    plt.plot([median + corr_x, median + corr_x], [0, y.max()], '--', color='red')
    plt.grid()
    plt.xlabel('Difference [um]')
    plt.ylabel('#')
    plt.yscale('log')
    plt.legend(loc=0)
    output_pdf.savefig()


def plot_alignments(x, mean_fitted, mean_error_fitted, n_cluster, ref_name, dut_name, prefix, non_interactive=False):
    '''PLots the correlation and lets the user cut on the data in an interactive way.

    Parameters
    ----------
    mean_fitted : array like
        The fitted peaks of one column / row correlation
    mean_error_fitted : array like
         The error of the fitted peaks of one column / row correlation
    n_cluster : array like
        The number of hits per column / row
    ref_name : string
        Reference name
    dut_name : string
        DUT name
    title : string
        Plot title
    non_interactive : boolean
        Deactivate user interaction to apply cuts
    '''
    # Global variables needed to manipulate them within a matplotlib QT slot function
    global selected_data
    global fit
    global do_refit
    global error_limit
    global offset_limit
    global left_limit
    global right_limit
    global x_diff
    global offset
    global fit
    global fit_fn

    x_diff = np.diff(x)[0]

    do_refit = True  # True as long as not the Refit button is pressed, needed to signal calling function that the fit is ok or not

    def update_offset(offset_limit_new):  # Function called when offset slider is moved
        global offset_limit
        offset_limit = offset_limit_new
        #offset_limit_plot.set_ydata([offset_limit, offset_limit])
        update_selected_data()
        update_plot()

    def update_error(error_limit_new):  # Function called when error slider is moved
        global error_limit
        error_limit = error_limit_new / 10.0
        #error_limit_plot.set_ydata([error_limit * 10.0, error_limit * 10.0])
        update_selected_data()
        update_plot()

    def update_left_limit(left_limit_new):  # Function called when left limit slider is moved
        global right_limit
        global left_limit
        global x_diff
        if left_limit_new >= right_limit - 1.0 * x_diff:
            left_limit_new = right_limit - 2.0 * x_diff
            left_slider.set_val(left_limit_new)
        left_limit = left_limit_new
        #left_limit_plot.set_xdata([left_limit, left_limit])
        update_selected_data()
        update_plot()

    def update_right_limit(right_limit_new):  # Function called when left limit slider is moved
        global right_limit
        global left_limit
        global x_diff
        if right_limit_new <= left_limit + 1.0 * x_diff:
            right_limit_new = left_limit + 2.0 * x_diff
            right_slider.set_val(right_limit_new)
        right_limit = right_limit_new
        #right_limit_plot.set_xdata([right_limit, right_limit])
        update_selected_data()
        update_plot()

    def update_selected_data():
        global selected_data
        global error_limit
        global offset_limit
        global left_limit
        global right_limit
        #init_selected_data()
        selected_data = initial_select.copy()
#         selected_data &= np.logical_and(np.logical_and(np.logical_and(np.abs(offset) <= offset_limit, np.abs(mean_error_fitted) <= error_limit), x >= left_limit), x <= right_limit)
        selected_data[selected_data] &= (np.abs(offset[selected_data]) <= offset_limit)
        selected_data[selected_data] &= (np.abs(mean_error_fitted[selected_data]) <= error_limit)
        selected_data &= (x >= left_limit)
        selected_data &= (x <= right_limit)

    def update_auto(event):  # Function called when auto button is pressed
        global selected_data
        global error_limit
        global offset_limit
        global left_limit
        global right_limit

        # This function automatically applies cuts according to these percentiles
        n_hit_percentile = 1
        mean_error_percentile = 95
        offset_percentile = 99

        error_median = np.nanmedian(mean_error_fitted[selected_data])
        error_std = np.nanstd(mean_error_fitted[selected_data])
        error_limit = max(error_median + error_std * 2, np.percentile(np.abs(mean_error_fitted[selected_data]), mean_error_percentile))
        offset_median = np.nanmedian(offset[selected_data])
        offset_std = np.nanstd(offset[selected_data])
        offset_limit = max(offset_median + offset_std * 2, np.percentile(np.abs(offset[selected_data]), offset_percentile))  # Do not cut too much on the offset, it depends on the fit that might be off

        n_hit_cut = np.percentile(n_cluster[selected_data], n_hit_percentile)  # Cut off low/high % of the hits
        n_hit_cut_index = np.zeros_like(n_cluster, dtype=np.bool)
        n_hit_cut_index |= (n_cluster < n_hit_cut)
        n_hit_cut_index[selected_data] |= (np.abs(offset[selected_data]) > offset_limit)
        n_hit_cut_index[~np.isfinite(offset)] = 1
        n_hit_cut_index[selected_data] |= (np.abs(mean_error_fitted[selected_data]) > error_limit)
        n_hit_cut_index[~np.isfinite(mean_error_fitted)] = 1
        n_hit_cut_index = np.where(n_hit_cut_index == 1)[0]
        left_index = np.where(x <= left_limit)[0][-1]
        right_index = np.where(x >= right_limit)[0][0]

        # update plot and selected data
        error_slider.set_val(error_limit * 10.0)
        offset_slider.set_val(offset_limit)
        n_hit_cut_index = n_hit_cut_index[n_hit_cut_index >= left_index]
        n_hit_cut_index = n_hit_cut_index[n_hit_cut_index <= right_index]
        if not np.any(n_hit_cut_index == left_index):
            n_hit_cut_index = np.r_[[left_index], n_hit_cut_index]
        if not np.any(n_hit_cut_index == right_index):
            n_hit_cut_index = np.r_[n_hit_cut_index, [right_index]]

        if np.any(n_hit_cut_index.shape):  # If data has no anomalies n_hit_cut_index is empty
            def consecutive(data, max_stepsize=10):  # Returns group of consecutive increasing values
                return np.split(data, np.where(np.diff(data) > max_stepsize)[0] + 1)
            cons = consecutive(n_hit_cut_index)
            left_cut = left_index if cons[0].shape[0] == 1 else cons[0][-1] + 1
            right_cut = right_index if cons[-1].shape[0] == 1 else cons[-1][0] - 1
            left_limit = x[left_cut]
            left_slider.set_val(left_limit)
            right_limit = x[right_cut]
            right_slider.set_val(right_limit)

        _fit_data()
        update_selected_data()
        update_plot()

    def update_plot():  # Replot correlation data with new selection
        global selected_data
        global offset
        if np.count_nonzero(selected_data) > 1:
            ax2.set_ylim(ymax=max(np.max(np.abs(mean_error_fitted[selected_data]) * 10.0), np.max(np.abs(offset[selected_data]))))
            offset_limit_plot.set_ydata([offset_limit, offset_limit])
            error_limit_plot.set_ydata([error_limit * 10.0, error_limit * 10.0])
            left_limit_plot.set_xdata([left_limit, left_limit])
            right_limit_plot.set_xdata([right_limit, right_limit])
            offset_plot.set_data(x[initial_select], np.abs(offset[initial_select]))
            offset_slider.valmax = np.max(np.abs(offset))
            offset_slider.ax.set_xlim(xmax=np.max(np.abs(offset)))
            mean_plot.set_data(x[selected_data], mean_fitted[selected_data])
            line_plot.set_data(x, fit_fn(x))
        else:
            if non_interactive:
                raise RuntimeError('Coarse alignment in non-interactive mode failed. Rerun with less iterations or in interactive mode!')
            else:
                logging.info('Cuts are too tight. Not enough data to fit')

    def init_selected_data():
        global selected_data
        selected_data = np.ones_like(mean_fitted, dtype=np.bool)
        selected_data &= np.isfinite(mean_fitted)
        selected_data &= np.isfinite(mean_error_fitted)

    def finish(event):  # Fit result is ok
        global do_refit
        do_refit = False  # Set to signal that no refit is required anymore
        plt.close()  # Close the plot to let the program continue (blocking)

    def refit(event):
        _fit_data()
        update_selected_data()
        update_plot()
        #plt.close()  # Close the plot to let the program continue (blocking)

    def _fit_data():
        global offset
        global fit
        global fit_fn

        fit, _ = curve_fit(testbeam_analysis.tools.analysis_utils.linear, x[selected_data], mean_fitted[selected_data])  # Fit straight line
        fit_fn = np.poly1d(fit[::-1])
        offset = fit_fn(x) - mean_fitted  # Calculate straight line fit offset
#         offset = np.full_like(mean_fitted, np.nan)
#         offset[selected_data] = fit_fn(x[selected_data]) - mean_fitted[selected_data]  # Calculate straight line fit offset

    # Require the gaussian fit error to be reasonable
#     selected_data = (mean_error_fitted < 1e-2)
    # Check for nan's and inf's
    init_selected_data()
    initial_select = selected_data.copy()
    # Calculate and plot selected data + fit + fit offset and gauss fit error
    plt.clf()
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()
    # Calculate and plot selected data + fit + fit offset and gauss fit error
    _fit_data()
    offset_limit = np.max(np.abs(offset[selected_data]))  # Calculate starting offset cut
    error_limit = np.max(np.abs(mean_error_fitted[selected_data]))  # Calculate starting fit error cut
    left_limit = np.min(x[selected_data])  # Calculate starting left cut
    right_limit = np.max(x[selected_data])  # Calculate starting right cut

    mean_plot, = ax.plot(x[selected_data], mean_fitted[selected_data], 'o-', label='Data prefit')  # Plot correlation
    line_plot, = ax.plot(x[selected_data], fit_fn(x[selected_data]), '-', label='Line fit')  # Plot line fit
    error_plot, = ax2.plot(x[selected_data], np.abs(mean_error_fitted[selected_data]) * 10.0, 'ro-', label='Error x10')  # Plot gaussian fit error
    offset_plot, = ax2.plot(x[selected_data], np.abs(offset[selected_data]), 'go-', label='Offset')  # Plot line fit offset
    offset_limit_plot = ax2.axhline(offset_limit, linestyle='--', color='g', linewidth=2)  # Plot offset cut as a line
    error_limit_plot = ax2.axhline(error_limit * 10.0, linestyle='--', color='r', linewidth=2)  # Plot error cut as a line
    left_limit_plot = ax2.axvline(left_limit, linestyle='-', color='b', linewidth=2)  # Plot left cut as a vertical line
    right_limit_plot = ax2.axvline(right_limit, linestyle='-', color='b', linewidth=2)  # Plot right cut as a vertical line
    ncluster_plot = ax.bar(x[selected_data], n_cluster[selected_data] / np.max(n_cluster[selected_data]).astype(np.float) * abs(np.diff(ax.get_ylim())[0]), bottom=ax.get_ylim()[0], align='center', alpha=0.1, label='#Cluster [a.u.]', width=np.min(np.diff(x[selected_data])))  # Plot number of hits for each correlation point
    ax.set_ylim(ymin=np.min(mean_fitted[selected_data]), ymax=np.max(mean_fitted[selected_data]))
    ax2.set_ylim(ymin=0.0)
    plt.xlim((np.nanmin(x), np.nanmax(x)))
    ax.set_title("Correlation of %s: %s vs. %s" % (prefix + "s", ref_name, dut_name))
    ax.set_xlabel("%s [um]" % dut_name)
    ax.set_ylabel("%s [um]" % ref_name)
    ax2.set_ylabel("Error / Offset")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    ax.grid()

    # Setup interactive sliders/buttons
    ax_offset = plt.axes([0.410, 0.04, 0.2, 0.02], axisbg='white')
    ax_error = plt.axes([0.410, 0.01, 0.2, 0.02], axisbg='white')
    ax_left_limit = plt.axes([0.125, 0.04, 0.2, 0.02], axisbg='white')
    ax_right_limit = plt.axes([0.125, 0.01, 0.2, 0.02], axisbg='white')
    ax_button_auto = plt.axes([0.670, 0.01, 0.06, 0.05], axisbg='black')
    ax_button_refit = plt.axes([0.735, 0.01, 0.08, 0.05], axisbg='black')
    ax_button_ok = plt.axes([0.82, 0.01, 0.08, 0.05], axisbg='black')
    # Create widgets
    offset_slider = Slider(ax_offset, 'Offset Cut', 0.0, offset_limit, valinit=offset_limit)
    error_slider = Slider(ax_error, 'Error cut', 0.0, error_limit * 10.0, valinit=error_limit * 10.0)
    left_slider = Slider(ax_left_limit, 'Left cut', left_limit, right_limit, valinit=left_limit)
    right_slider = Slider(ax_right_limit, 'Right cut', left_limit, right_limit, valinit=right_limit)
    auto_button = Button(ax_button_auto, 'Auto')
    refit_button = Button(ax_button_refit, 'Refit')
    ok_button = Button(ax_button_ok, 'OK')
    # Connect slots
    offset_slider.on_changed(update_offset)
    error_slider.on_changed(update_error)
    left_slider.on_changed(update_left_limit)
    right_slider.on_changed(update_right_limit)
    auto_button.on_clicked(update_auto)
    refit_button.on_clicked(refit)
    ok_button.on_clicked(finish)

    if non_interactive:
        update_auto(None)
    else:
        plt.get_current_fig_manager().window.showMaximized()  # Plot needs to be large, so maximize
        plt.show()

    return selected_data, fit, do_refit  # Return cut data for further processing


def plot_alignment_fit(x, mean_fitted, mask, fit_fn, fit, pcov, chi2, mean_error_fitted, n_cluster, n_pixel_ref, n_pixel_dut, pixel_size_ref, pixel_size_dut, ref_name, dut_name, prefix, output_pdf):
    plt.clf()
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()
    ax.errorbar(x[mask], mean_fitted[mask], yerr=mean_error_fitted[mask], linestyle='', color="blue", fmt='.', label='Correlation', zorder=10)
    ax2.plot(x[mask], mean_error_fitted[mask] * 10.0, linestyle='', color="red", marker='o', label='Error x10')
    ax2.errorbar(x[mask], np.abs(fit_fn(x[mask]) - mean_fitted[mask]), mean_error_fitted[mask], linestyle='', color="lightgreen", marker='o', label='Offset')
    ax2.plot(x, chi2 / 1e5, 'r--', label="Chi$^2$")
    # Plot masked data points, but they should not influence the ylimit
    y_limits = ax2.get_ylim()
    plt.errorbar(x[~mask], mean_fitted[~mask], yerr=mean_error_fitted[~mask], linestyle='', color="darkblue", fmt='.')
    plt.plot(x[~mask], mean_error_fitted[~mask] * 10.0, linestyle='', color="darkred", marker='o')
    plt.errorbar(x[~mask], np.abs(fit_fn(x[~mask]) - mean_fitted[~mask]), mean_error_fitted[~mask], linestyle='', color="darkgreen", marker='o')
    ax2.set_ylim(y_limits)
    ax2.set_ylim(ymin=0.0)
    ax.set_ylim((-n_pixel_ref * pixel_size_ref / 2.0, n_pixel_ref * pixel_size_ref / 2.0))
    plt.xlim((-n_pixel_dut * pixel_size_dut / 2.0, n_pixel_dut * pixel_size_dut / 2.0))
    ax2.bar(x, n_cluster / np.max(n_cluster).astype(np.float) * ax2.get_ylim()[1], align='center', alpha=0.1, label='#Cluster [a.u.]', width=np.min(np.diff(x)))  # Plot number of hits for each correlation point
    # Plot again to draw line above the markers
    if len(pcov) > 1:
        fit_legend_entry = 'Fit: $c_0+c_1*x$\n$c_0=%.1e \pm %.1e$\n$c_1=%.1e \pm %.1e$' % (fit[0], np.absolute(pcov[0][0]) ** 0.5, fit[1], np.absolute(pcov[1][1]) ** 0.5)
    else:
        fit_legend_entry = 'Fit: $c_0+x$\n$c_0=%.1e \pm %.1e$' % (fit[0], np.absolute(pcov[0][0]) ** 0.5)
    ax.plot(x, fit_fn(x), linestyle='-', color="darkorange", label=fit_legend_entry)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.title("Correlation of %s: %s vs. %s" % (prefix + "s", ref_name, dut_name))
    ax.set_xlabel("%s [um]" % dut_name)
    ax.set_ylabel("%s [um]" % ref_name)
    ax2.set_ylabel("Error [a.u.]")
#     plt.xlim((0, x.shape[0]))
    ax.grid()
    if output_pdf:
        output_pdf.savefig()
    else:
        plt.show()


def plot_hough(x, data, accumulator, offset, slope, theta_edges, rho_edges, n_pixel_ref, n_pixel_dut, pixel_size_ref, pixel_size_dut, ref_name, dut_name, prefix, output_pdf):
    capital_prefix = prefix
    if prefix:
        capital_prefix = prefix.title()
    if pixel_size_dut and pixel_size_ref:
        aspect = pixel_size_ref / pixel_size_dut
    else:
        aspect = "auto"

    plt.clf()
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1)
    cmap = cm.get_cmap('viridis')
    plt.imshow(accumulator, interpolation="none", origin="lower", aspect="auto", cmap=cmap, extent=[np.rad2deg(theta_edges[0]), np.rad2deg(theta_edges[-1]), rho_edges[0], rho_edges[-1]])
    plt.xticks([-90, -45, 0, 45, 90])
    plt.title("Accumulator plot of %s correlations: %s vs. %s" % (prefix, ref_name, dut_name))
    ax.set_xlabel(r'$\theta$ [degree]')
    ax.set_ylabel(r'$\rho$ [um]')
    if output_pdf:
        output_pdf.savefig()
    else:
        plt.show()
    plt.clf()
    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1)
    fit_legend_entry = 'Hough: $c_0+c_1*x$\n$c_0=%.1e$\n$c_1=%.1e$' % (offset, slope)
    plt.plot(x, testbeam_analysis.tools.analysis_utils.linear(x, offset, slope), linestyle=':', color="darkorange", label=fit_legend_entry)
    plt.imshow(data, interpolation="none", origin="lower", aspect=aspect, cmap='Greys')
    plt.title("Correlation of %s: %s vs. %s" % (prefix + "s", ref_name, dut_name))
    ax.set_xlabel("%s %s" % (capital_prefix, dut_name))
    ax.set_ylabel("%s %s" % (capital_prefix, ref_name))
    ax.legend(loc=0)
    if output_pdf:
        output_pdf.savefig()
    else:
        plt.show()

def plot_correlations(input_correlation_file, output_pdf_file=None, pixel_size=None, dut_names=None):
    '''Takes the correlation histograms and plots them

    Parameters
    ----------
    input_correlation_file : pytables file
        The input file with the correlation histograms.
    output_pdf_file : pdf file name
    '''

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_correlation_file)[0] + '.pdf'

    with PdfPages(output_pdf_file) as output_pdf:
        with tb.open_file(input_correlation_file, mode="r") as in_file_h5:
            for node in in_file_h5.root:
                try:
                    indices = re.findall(r'\d+', node.name)
                    dut_idx = int(indices[0])
                    ref_idx = int(indices[1])
                    if "column" in node.name.lower():
                        column = True
                    else:
                        column = False
                except AttributeError:
                    continue
                data = node[:]

                if np.all(data <= 0):
                    logging.warning('All correlation entries for %s are zero, do not create plots', str(node.name))
                    continue

                plt.clf()
                cmap = cm.get_cmap('viridis')
#                 cmap.set_bad('w')
                norm = colors.LogNorm()
                if pixel_size:
                    aspect = pixel_size[ref_idx][0 if column else 1] / (pixel_size[dut_idx][0 if column else 1])
                else:
                    aspect = "auto"
                im = plt.imshow(data.T, origin="lower", cmap=cmap, norm=norm, aspect=aspect, interpolation='none')
                dut_name = dut_names[dut_idx] if dut_names else ("DUT " + str(dut_idx))
                ref_name = dut_names[ref_idx] if dut_names else ("DUT " + str(ref_idx))
                plt.title("Correlation of %s: %s vs. %s" % ("columns" if "column" in node.title.lower() else "rows", dut_name, ref_name))
                plt.xlabel('%s %s' % ("Column" if "column" in node.title.lower() else "Row", dut_name))
                plt.ylabel('%s %s' % ("Column" if "column" in node.title.lower() else "Row", ref_name))
                # do not append to axis to preserve aspect ratio
                plt.colorbar(im, cmap=cmap, norm=norm, fraction=0.04, pad=0.05)
#                 divider = make_axes_locatable(plt.gca())
#                 cax = divider.append_axes("right", size="5%", pad=0.1)
#                 z_max = np.amax(data)
#                 plt.colorbar(im, cax=cax, ticks=np.linspace(start=0, stop=z_max, num=9, endpoint=True))
                output_pdf.savefig()


def plot_hit_alignment(title, difference, particles, ref_dut_column, table_column, actual_median, actual_mean, output_fig, bins=100):
    plt.clf()
    plt.hist(difference, bins=bins, range=(-1. / 100. * np.amax(particles[:][ref_dut_column]) / 1., 1. / 100. * np.amax(particles[:][ref_dut_column]) / 1.))
    try:
        plt.yscale('log')
    except ValueError:
        pass
    plt.xlabel('%s - %s' % (ref_dut_column, table_column))
    plt.ylabel('#')
    plt.title(title)
    plt.grid()
    plt.plot([actual_median, actual_median], [0, plt.ylim()[1]], '-', linewidth=2.0, label='Median %1.1f' % actual_median)
    plt.plot([actual_mean, actual_mean], [0, plt.ylim()[1]], '-', linewidth=2.0, label='Mean %1.1f' % actual_mean)
    plt.legend(loc=0)
    output_fig.savefig()


def plot_hit_alignment_2(in_file_h5, combine_n_hits, median, mean, correlation, alignment, output_fig):
    plt.clf()
    plt.xlabel('Hits')
    plt.ylabel('Offset')
    plt.grid()
    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_hits), median, linewidth=2.0, label='Median')
    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_hits), mean, linewidth=2.0, label='Mean')
    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_hits), correlation, linewidth=2.0, label='Alignment')
    plt.legend(loc=0)
    output_fig.savefig()


def plot_z(z, dut_z_col, dut_z_row, dut_z_col_pos_errors, dut_z_row_pos_errors, dut_index, output_fig):
    plt.clf()
    plt.plot([dut_z_col.x, dut_z_col.x], [0., 1.], "--", label="DUT%d, col, z=%1.4f" % (dut_index, dut_z_col.x))
    plt.plot([dut_z_row.x, dut_z_row.x], [0., 1.], "--", label="DUT%d, row, z=%1.4f" % (dut_index, dut_z_row.x))
    plt.plot(z, dut_z_col_pos_errors / np.amax(dut_z_col_pos_errors), "-", label="DUT%d, column" % dut_index)
    plt.plot(z, dut_z_row_pos_errors / np.amax(dut_z_row_pos_errors), "-", label="DUT%d, row" % dut_index)
    plt.grid()
    plt.legend(loc=1)
    plt.ylim((np.amin(dut_z_col_pos_errors / np.amax(dut_z_col_pos_errors)), 1.))
    plt.xlabel('Relative z-position')
    plt.ylabel('Mean squared offset [a.u.]')
    plt.gca().set_yscale('log')
    plt.gca().get_yaxis().set_ticks([])
    output_fig.savefig()


def plot_events(input_tracks_file, event_range, dut=None, max_chi2=None, output_pdf=None):
    '''Plots the tracks (or track candidates) of the events in the given event range.

    Parameters
    ----------
    input_tracks_file : pytables file with tracks
    event_range : iterable:
        (start event number, stop event number(
    dut : int
        Take data from this DUT
    max_chi2 : int
        Plot only converged fits (cut on chi2)
    output_pdf : pdf file name
    '''

    output_fig = PdfPages(output_pdf) if output_pdf else None

    with tb.open_file(input_tracks_file, "r") as in_file_h5:
        fitted_tracks = False
        try:  # data has track candidates
            table = in_file_h5.root.TrackCandidates
        except tb.NoSuchNodeError:  # data has fitted tracks
            table = in_file_h5.get_node(in_file_h5.root, name='Tracks_DUT_%d' % dut)
            fitted_tracks = True

        n_duts = sum(['charge' in col for col in table.dtype.names])
        array = table[:]
        tracks = testbeam_analysis.tools.analysis_utils.get_data_in_event_range(array, event_range[0], event_range[-1])
        if tracks.shape[0] == 0:
            logging.warning('No tracks in event selection, cannot plot events!')
            return
        if max_chi2:
            tracks = tracks[tracks['track_chi2'] <= max_chi2]
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for track in tracks:
            x, y, z = [], [], []
            for dut_index in range(0, n_duts):
                if track['x_dut_%d' % dut_index] != 0:  # No hit has x = 0
                    x.append(track['x_dut_%d' % dut_index] * 1.e-3)  # in mm
                    y.append(track['y_dut_%d' % dut_index] * 1.e-3)  # in mm
                    z.append(track['z_dut_%d' % dut_index] * 1.e-3)  # in mm

            if fitted_tracks:
                offset = np.array((track['offset_0'], track['offset_1'], track['offset_2']))
                slope = np.array((track['slope_0'], track['slope_1'], track['slope_2']))
                linepts = offset * 1.e-3 + slope * 1.e-3 * np.mgrid[-150000:150000:2000j][:, np.newaxis]

            n_hits = bin(track['track_quality'] & 0xFF).count('1')
            n_very_good_hits = bin(track['track_quality'] & 0xFF0000).count('1')

            if n_hits > 2:  # only plot tracks with more than 2 hits
                if fitted_tracks:
                    ax.plot(x, y, z, '.' if n_hits == n_very_good_hits else 'o')
                    ax.plot3D(*linepts.T)
                else:
                    ax.plot(x, y, z, '.-' if n_hits == n_very_good_hits else '.--')

#         ax.set_xlim(0, 20)
#         ax.set_ylim(0, 20)
        ax.set_zlim(np.amin(np.array(z)), np.amax(np.array(z)))
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('z [mm]')
        plt.title('%d tracks of %d events' % (tracks.shape[0], np.unique(tracks['event_number']).shape[0]))
        if output_pdf is not None:
            output_fig.savefig()
        else:
            plt.show()

    if output_fig:
        output_fig.close()


def plot_track_chi2(chi2s, fit_dut, output_fig):
    # Plot track chi2 and angular distribution
    plt.clf()
    limit_x = np.percentile(chi2s, q=68.3)  # Plot up to 1 sigma of the chi2 range
    plt.hist(chi2s, bins=200, range=(0, limit_x))
    plt.grid()
    plt.xlabel('Track Chi2 [um*um]')
    plt.ylabel('#')
    plt.yscale('log')
    plt.title('Track Chi2 for DUT %d tracks' % fit_dut)
    output_fig.savefig()


def plot_residuals(histogram, edges, fit, fit_errors, x_label, title, output_fig=None):
    for plot_log in [False, True]:  # plot with log y or not
        plt.clf()
        # Calculate bin centers
        x = (edges[1:] + edges[:-1]) / 2.0
        plot_range = (testbeam_analysis.tools.analysis_utils.get_mean_from_histogram(histogram, x) - 5 * testbeam_analysis.tools.analysis_utils.get_rms_from_histogram(histogram, x),
                      testbeam_analysis.tools.analysis_utils.get_mean_from_histogram(histogram, x) + 5 * testbeam_analysis.tools.analysis_utils.get_rms_from_histogram(histogram, x))
        plt.xlim(plot_range)
        plt.grid()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel('#')

        if plot_log:
            plt.ylim(1, int(ceil(np.amax(histogram) / 10.0)) * 100)

        plt.bar(x, histogram, log=plot_log, align='center')
        if np.any(fit):
            plt.plot([fit[1], fit[1]], [0, plt.ylim()[1]], color='red', label='Entries %d\nRMS %.1f um' % (histogram.sum(), testbeam_analysis.tools.analysis_utils.get_rms_from_histogram(histogram, x)))
            gauss_fit_legend_entry = 'Gauss fit: \nA=$%.1f\pm %.1f$\nmu=$%.1f\pm %.1f$\nsigma=$%.1f\pm %.1f$' % (fit[0], np.absolute(fit_errors[0][0] ** 0.5), fit[1], np.absolute(fit_errors[1][1] ** 0.5), np.absolute(fit[2]), np.absolute(fit_errors[2][2] ** 0.5))
            x_gauss = np.arange(np.floor(np.min(edges)), np.ceil(np.max(edges)), step=0.1)
            plt.plot(x_gauss, testbeam_analysis.tools.analysis_utils.gauss(x_gauss, *fit), 'r--', label=gauss_fit_legend_entry, linewidth=2)
            plt.legend(loc=0)
        plt.xlim([edges[0], edges[-1]])
        if output_fig:
            output_fig.savefig()
        elif output_fig is None:
            plt.show()


def plot_position_residuals(hist, xedges, yedges, x, y, x_label, y_label, title=None, yerr=None, output_fig=None, fit=None):  # Plot the residuals as a function of the position
    plt.clf()
    plt.grid()
    if title:
        plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.imshow(np.ma.masked_equal(hist, 0).T, origin='low', aspect='auto', interpolation='none', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], label='Residual')
    plt.plot(x, y, 'go', linewidth=2, label='Median residual')
    if fit is not None:
        axis = plt.gca()
        x_lim = np.array(axis.get_xlim(), dtype=np.float)
        plt.plot(x_lim, testbeam_analysis.tools.analysis_utils.linear(x_lim, *fit[0]), linestyle='-', color="darkorange", linewidth=2, label='Mean residual fit\n%.2e + %.2e x' % (fit[0][0], fit[0][1]))
    plt.legend(loc=0)
    if output_fig:
        output_fig.savefig()
    elif output_fig is None:
        plt.show()


def plot_residuals_vs_position(hist, xedges, yedges, xlabel, ylabel, res_mean=None, res_pos=None, selection=None, title=None, output_fig=None, fit=None, cov=None):  # Plot the residuals as a function of the position
    plt.clf()
    plt.grid()
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.imshow(np.ma.masked_equal(hist, 0).T, extent=[xedges[0], xedges[-1] , yedges[0], yedges[-1]], origin='low', aspect='auto', interpolation='none')
    if res_mean is not None and res_pos is not None:
        if selection is None:
            selection = np.full_like(res_pos, True, dtype=np.bool)
        plt.plot(res_pos[selection], res_mean[selection], linestyle='', color="blue", marker='o',  label='Mean residual')
        plt.plot(res_pos[~selection], res_mean[~selection], linestyle='', color="darkblue", marker='o')
    if fit is not None:
        axis = plt.gca()
        x_lim = np.array(axis.get_xlim(), dtype=np.float)
        plt.plot(x_lim, testbeam_analysis.tools.analysis_utils.linear(x_lim, *fit), linestyle='-', color="darkorange", linewidth=2, label='Mean residual fit\n%.2e + %.2e x' % (fit[0], fit[1]))
    plt.xlim([xedges[0], xedges[-1]])
    plt.ylim([yedges[0], yedges[-1]])
    plt.legend(loc=0)
    if output_fig:
        output_fig.savefig()
    elif output_fig is None:
        plt.show(),


def plot_track_density(input_tracks_file, output_pdf, z_positions, dim_x, dim_y, pixel_size, mask_zero=True, use_duts=None, max_chi2=None):
    '''Takes the tracks and calculates the track density projected on selected DUTs.
    Parameters
    ----------
    input_tracks_file : string
        file name with the tracks table
    output_pdf : pdf file name object
    z_positions : iterable
        Iterable with z-positions of all DUTs
    dim_x, dim_y : integer
        front end dimensions of device (number of pixel)
    pixel_size : iterable
        pixel size (x, y) for every plane
    mask_zero : bool
        Mask heatmap entries = 0 for plotting
    use_duts : iterable
        the duts to plot track density for. If None all duts are used
    max_chi2 : int
        only use tracks with a chi2 <= max_chi2
    '''
    logging.info('Plot track density')
    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            plot_ref_dut = False
            dimensions = []

            for index, node in enumerate(in_file_h5.root):
                # Bins define (virtual) pixel size for histogramming
                bin_x, bin_y = dim_x, dim_y

                # Calculate dimensions in um for every plane
                dimensions.append((dim_x * pixel_size[index][0], dim_y * pixel_size[index][1]))

                plot_range = (dimensions[index][0], dimensions[index][1])

                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.info('Plot track density for DUT %d', actual_dut)

                track_array = node[:]

                # If set, select only converged fits
                if max_chi2:
                    track_array = track_array[track_array['track_chi2'] <= max_chi2]

                if plot_ref_dut:  # Plot first and last device
                    heatmap_ref_hits, _, _ = np.histogram2d(track_array['column_dut_0'], track_array['row_dut_0'], bins=(bin_x, bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])
                    if mask_zero:
                        heatmap_ref_hits = np.ma.array(heatmap_ref_hits, mask=(heatmap_ref_hits == 0))

                    # Get number of hits in DUT0
                    n_ref_hits = np.count_nonzero(heatmap_ref_hits)

                    fig = Figure()
                    fig.patch.set_facecolor('white')
                    ax = fig.add_subplot(111)
                    plot_2d_pixel_hist(fig, ax, heatmap_ref_hits.T, plot_range, title='Hit density for DUT 0 (%d Hits)' % n_ref_hits, x_axis_title="column [um]", y_axis_title="row [um]")
                    fig.tight_layout()
                    output_fig.savefig(fig)

                    plot_ref_dut = False

                offset, slope = np.column_stack((track_array['offset_0'], track_array['offset_1'], track_array['offset_2'])), np.column_stack((track_array['slope_0'], track_array['slope_1'], track_array['slope_2']))
                intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[actual_dut] - offset[:, 2, np.newaxis])  # intersection track with DUT plane

                heatmap, _, _ = np.histogram2d(intersection[:, 0], intersection[:, 1], bins=(bin_x, bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])
                heatmap_hits, _, _ = np.histogram2d(track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], bins=(bin_x, bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])

                # For better readability allow masking of entries that are zero
                if mask_zero:
                    heatmap = np.ma.array(heatmap, mask=(heatmap == 0))
                    heatmap_hits = np.ma.array(heatmap_hits, mask=(heatmap_hits == 0))

                # Get number of hits / tracks
                n_hits_heatmap = np.count_nonzero(heatmap)
                n_hits_heatmap_hits = np.count_nonzero(heatmap_hits)

                fig = Figure()
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111)
                plot_2d_pixel_hist(fig, ax, heatmap.T, plot_range, title='Track density for DUT %d tracks (%d Tracks)' % (actual_dut, n_hits_heatmap), x_axis_title="column [um]", y_axis_title="row [um]")
                fig.tight_layout()
                output_fig.savefig(fig)

                fig = Figure()
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111)
                plot_2d_pixel_hist(fig, ax, heatmap_hits.T, plot_range, title='Hit density for DUT %d (%d Hits)' % (actual_dut, n_hits_heatmap_hits), x_axis_title="column [um]", y_axis_title="row [um]")
                fig.tight_layout()
                output_fig.savefig(fig)


def plot_charge_distribution(input_track_candidates_file, output_pdf, dim_x, dim_y, pixel_size, mask_zero=True, use_duts=None):
    '''Takes the data and plots the charge distribution for selected DUTs.
    Parameters
    ----------
    input_track_candidates_file : string
        file name with the tracks table
    output_pdf : pdf file name object
    dim_x, dim_y : integer
        front end dimensions of device (number of pixel)
    pixel_size : iterable
        pixel size (x, y) for every plane
    mask_zero : bool
        Mask heatmap entries = 0 for plotting
    use_duts : iterable
        the duts to plot track density for. If None all duts are used
    '''
    logging.info('Plot charge distribution')
    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
            dimensions = []
            for table_column in in_file_h5.root.TrackCandidates.dtype.names:
                if 'charge' in table_column:
                    actual_dut = int(re.findall(r'\d+', table_column)[-1])
                    index = actual_dut

                    # allow one channel value for all planes or one value for each plane
                    channels_x = [dim_x, ] if not isinstance(dim_x, tuple) else dim_x
                    channels_y = [dim_y, ] if not isinstance(dim_y, tuple) else dim_y
                    if len(channels_x) == 1:  # if one value for all planes
                        n_bin_x, n_bin_y = channels_x, channels_y  # Bins define (virtual) pixel size for histogramming
                        dimensions.append((channels_x * pixel_size[index][0], channels_y * pixel_size[index][1]))  # Calculate dimensions in um for every plane

                    else:  # if one value for each plane
                        n_bin_x, n_bin_y = channels_x[index], channels_y[index]  # Bins define (virtual) pixel size for histogramming
                        dimensions.append((channels_x[index] * pixel_size[index][0], channels_y[index] * pixel_size[index][1]))  # Calculate dimensions in um for every plane

                    plot_range = (dimensions[index][0], dimensions[index][1])

                    if use_duts and actual_dut not in use_duts:
                        continue
                    logging.info('Plot charge distribution for DUT %d', actual_dut)

                    track_array = in_file_h5.root.TrackCandidates[:]

                    n_bins_charge = int(np.amax(track_array['charge_dut_%d' % actual_dut]))

                    x_y_charge = np.column_stack((track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], track_array['charge_dut_%d' % actual_dut]))
                    hit_hist, _, _ = np.histogram2d(track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], bins=(n_bin_x, n_bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])
                    charge_distribution = np.histogramdd(x_y_charge, bins=(n_bin_x, n_bin_y, n_bins_charge), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5], [0, n_bins_charge]])[0]

                    charge_density = np.average(charge_distribution, axis=2, weights=range(0, n_bins_charge)) * sum(range(0, n_bins_charge)) / hit_hist.astype(float)
                    charge_density = np.ma.masked_invalid(charge_density)

                    fig = Figure()
                    fig.patch.set_facecolor('white')
                    ax = fig.add_subplot(111)
                    plot_2d_pixel_hist(fig, ax, charge_density.T, plot_range, title='Charge density for DUT %d' % actual_dut, x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=int(np.ma.average(charge_density) * 1.5))
                    fig.tight_layout()
                    output_fig.savefig(fig)

def plot_track_distances(distance_min_array, distance_max_array, distance_mean_array, actual_dut, plot_range, cut_distance, output_fig):
    # get number of entries for every histogram
    n_hits_distance_min_array = distance_min_array.count()
    n_hits_distance_max_array = distance_max_array.count()
    n_hits_distance_mean_array = distance_mean_array.count()
    
    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, distance_min_array.T, plot_range, title='Minimal distance for DUT %d (%d Hits)' % (actual_dut, n_hits_distance_min_array), x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=125000)
    fig.tight_layout()
    output_fig.savefig(fig)
    
    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, distance_max_array.T, plot_range, title='Maximal distance for DUT %d (%d Hits)' % (actual_dut, n_hits_distance_max_array), x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=125000)
    fig.tight_layout()
    output_fig.savefig(fig)

    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, distance_mean_array.T, plot_range, title='Weighted distance for DUT %d (%d Hits)' % (actual_dut, n_hits_distance_mean_array), x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=cut_distance)
    fig.tight_layout()
    output_fig.savefig(fig)


def efficiency_plots(hit_hist, track_density, track_density_with_DUT_hit, efficiency, actual_dut, minimum_track_density, plot_range, cut_distance, output_fig, mask_zero=True):
    # get number of entries for every histogram
    n_hits_hit_hist = np.count_nonzero(hit_hist)
    n_tracks_track_density = np.count_nonzero(track_density)
    n_tracks_track_density_with_DUT_hit = np.count_nonzero(track_density_with_DUT_hit)
    n_hits_efficiency = np.count_nonzero(efficiency)

    # for better readability allow masking of entries that are zero
    if mask_zero:
        hit_hist = np.ma.array(hit_hist, mask=(hit_hist == 0))
        track_density = np.ma.array(track_density, mask=(track_density == 0))
        track_density_with_DUT_hit = np.ma.array(track_density_with_DUT_hit, mask=(track_density_with_DUT_hit == 0))

    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, hit_hist.T, plot_range, title='Hit density for DUT %d (%d Hits)' % (actual_dut, n_hits_hit_hist), x_axis_title="column [um]", y_axis_title="row [um]")
    fig.tight_layout()
    output_fig.savefig(fig)

    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, track_density.T, plot_range, title='Track density for DUT %d (%d Tracks)' % (actual_dut, n_tracks_track_density), x_axis_title="column [um]", y_axis_title="row [um]")
    fig.tight_layout()
    output_fig.savefig(fig)

    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, track_density_with_DUT_hit.T, plot_range, title='Density of tracks with DUT hit for DUT %d (%d Tracks)' % (actual_dut, n_tracks_track_density_with_DUT_hit), x_axis_title="column [um]", y_axis_title="row [um]")
    fig.tight_layout()
    output_fig.savefig(fig)

    if np.any(~efficiency.mask):
        fig = Figure()
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        z_min = np.ma.min(efficiency)
        if z_min == 100.:  # One cannot plot with 0 z axis range
            z_min = 90.
        plot_2d_pixel_hist(fig, ax, efficiency.T, plot_range, title='Efficiency for DUT %d (%d Entries)' % (actual_dut, n_hits_efficiency), x_axis_title="column [um]", y_axis_title="row [um]", z_min=z_min, z_max=100.)
        fig.tight_layout()
        output_fig.savefig(fig)

        plt.clf()
        plt.grid()
        plt.title('Efficiency per pixel for DUT %d: %1.4f +- %1.4f' % (actual_dut, np.ma.mean(efficiency), np.ma.std(efficiency)))
        plt.xlabel('Efficiency [%]')
        plt.ylabel('#')
        plt.yscale('log')
        plt.xlim([-0.5, 101.5])
        plt.hist(efficiency.ravel()[efficiency.ravel().mask != 1], bins=101, range=(0, 100))  # Histogram not masked pixel efficiency
        output_fig.savefig()
    else:
        logging.warning('Cannot create efficiency plots, since all pixels are masked')
