# @ ImagePlus imp

from ij import *
from ij import IJ
from ij.gui import Roi, PointRoi, GenericDialog, Overlay
from ij.plugin import Duplicator
from ij.plugin.frame import RoiManager
from ij.io import FileSaver
from ConfigParser import SafeConfigParser
from datetime import datetime
import subprocess
import shlex
import os
import glob
import csv
import math


def error_dialog(title, message):
    """ Creates a dialog box with a message and throws an exception

    Args:
        title::str
            The title at the top of the dialog box
        message::str
            The message within the dialog box and in the exception

    Returns:
        Dialog box with message
    """
    gui = GenericDialog(title)
    gui.addMessage(message)
    gui.hideCancelButton()
    gui.showDialog()
    raise Exception(message)

def adjust_count(input_count, slope, intercept):
	""" Adjusts cell counts based on linear equation

	Args:
		input_count::int
			The number of cells in a given ROI
		slope::float
			The slope of the linear equation to adjust with
		intercept::float
			The intercept of the linear equation to adjust with
			
	Returns:
		Float of number of cells in an ROI
	"""
	return (input_count-intercept)/float(slope)

def run_script():
    # Tile Selection Portion
    # Creates a new ImagePlus
    imp = IJ.getImage()
    width = imp.getDimensions()[0]
    height = imp.getDimensions()[1]
    
    imp_overlay = imp.getOverlay()

    # Gets number of tiles selected
    number_tiles = imp_overlay.size()

    # Checks if points have been selected, displays warning and raises exception if not
    if imp_overlay is None:
        error_dialog('Warning', 'Select some points first and re-run the plugin.')

    # Iterator for points (as points are not usually iterable)
    iterate = imp_overlay.iterator()

    # Config setup
    # TODO - Double check path works on Windows 
    plugin_wd = IJ.getDirectory('current') + 'scripts/'
    IJ.log(plugin_wd)
    config = SafeConfigParser()
    config.read(plugin_wd + 'plugin_config.ini')
    python_path = config.get('path_info','python_path')
    logging_bool = config.getboolean('path_info','logging')
    temp_folder = plugin_wd + "temp/"

    # Parameter Dialog
    parameter_dialog = GenericDialog("Set Parameters")
    parameter_dialog.addNumericField("Box Size (px)", 272)
    parameter_dialog.addSlider("Splitting Threshold", 0, 100, 40)
    parameter_dialog.addSlider("Edge Threshold (bottom & left)", 0, 200, 10)
    parameter_dialog.hideCancelButton()
    parameter_dialog.showDialog() 
    
    box_size = parameter_dialog.getNextNumber()
    sthres_val = parameter_dialog.getSliders().get(0).getValue() 
    edgethres_val = parameter_dialog.getSliders().get(1).getValue() 
    safe_zone = box_size / 2

    i = 1  # Counter
    all_counts = []
    while iterate.hasNext():
        IJ.showStatus("Processing region " + str(i) + "/" + str(number_tiles))

        p = iterate.next()

        # Uncomment to print width and height of image to console
        # print(p.getFloatWidth())
        # print(p.getFloatHeight())

        # Uncomment to print centroid of ROI to console
        # print(p.getContourCentroid()[0])
        # print(p.getContourCentroid()[1])

        if safe_zone < p.getContourCentroid()[0] < (width - safe_zone):
            x_min = p.getContourCentroid()[0] - safe_zone  # Calculation for correct constraint selection
        else:
            if p.getContourCentroid()[0] < safe_zone:
                x_min = 0  # If selected point is outside safe selection (100 px from edge)
            else:
                x_min = width - box_size

        if safe_zone < p.getContourCentroid()[1] < (height - safe_zone):
            y_min = p.getContourCentroid()[1] - safe_zone  # Calculation for correct constraint selection
        else:
            if p.getContourCentroid()[1] < safe_zone:
                y_min = 0  # If selected point is outside safe selection (100 px from edge)
            else:
                y_min = height - box_size

        bounds = Roi(x_min, y_min, box_size, box_size)  # Creating ROI with correct 200x200 bounds

        cropped_imp = Duplicator().run(imp)
        no_contrast = Duplicator().run(imp)
        cropped_imp.setRoi(bounds)  # Setting cropping boundaries to bounds ROI
        no_contrast.setRoi(bounds)
        IJ.run(cropped_imp, 'Crop', '')
        IJ.run(no_contrast, 'Crop', '')
        IJ.run(cropped_imp, "Enhance Contrast", "saturated=0.35")  # automatic contrast
        cropped_imp.setTitle('region_' + str(i))  # Naming image sequentially
        no_contrast.setTitle('unfiltered_region_' + str(i)) 

        # Saves images as temporary files to be passed into algorithm subprocess
        output_file = FileSaver(cropped_imp)
        output_file.saveAsJpeg(temp_folder + cropped_imp.getTitle() + ".jpg")
        output_filename = (temp_folder + cropped_imp.getTitle() + ".jpg")
        output_filename = '"{}"'.format(output_filename)

        IJ.showStatus("Identifying cells in region " + str(i) + "/" + str(number_tiles))

        shell_in = python_path + " Count_cells_V2_radius.py " + output_filename + " --sthres " + str(sthres_val) + " --cutradius " + str(edgethres_val)
        args = shlex.split(shell_in)

        # Calls cell detection algorithm and saves output as string
        try:
            process = subprocess.check_output(args, stderr=subprocess.STDOUT, shell=False,
                                              cwd=plugin_wd)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        IJ.showStatus("Plotting cell locations in region " + str(i) + "/" + str(number_tiles))

        # Point Selection Portion
        all_points = RoiManager(False)  # Creates RoiManager to store points and hides it

        inset_overlay = Overlay()  # Not needed if using RoiManager
        no_contrast_overlay = Overlay()
        cropped_imp.setOverlay(inset_overlay)  # Not needed if using RoiManager
        no_contrast.setOverlay(no_contrast_overlay)

        ROI_output = []  # List of ROIs to print to log as [x_coord, y_coord]

        # Reads output from subprocess and sets points as ROIs
        process = process.split('\n')
        num_points = 0
        for line in process:
            if len(line) != 0:
                print(line)
                line = line.split(",")
                x_coord = int(line[0])  # Sets x-coordinate
                y_coord = int(line[1])  # Sets y-coordinate
                ROI_output.append([x_coord, y_coord])
                new_pt = PointRoi(x_coord, y_coord)
                print(new_pt)
                new_pt.setPointType(0)
                new_pt.setName('%04d-%04d' % (y_coord, x_coord))
                all_points.add(cropped_imp, new_pt, -1)
                num_points += 1

        all_points.moveRoisToOverlay(cropped_imp)
        all_points.moveRoisToOverlay(no_contrast)
        # inset_overlay.drawNames(True)  # Uncomment this to draw labels instead of numbers
        cropped_imp.show()  # Displays inset image 
        no_contrast.show()

        print("Number of points is " + str(num_points))
        all_counts.append(num_points)

        # Automatic density calculation
        dt_now = datetime.utcnow()
        dt_string = dt_now.strftime("%y%m%d%H%M%S")
        IJ.showStatus("Calculating density for region " + str(i) + "/" + str(number_tiles))
        density_calc = (1000 ** 2 * num_points) / 40000
        IJ.log("Region " + str(i) + " cell count: " + str(num_points))
        adjusted_count = adjust_count(num_points, 0.4185, 88.9998)
        IJ.log("Region " + str(i) + " ADJUSTED cell count: %.2f" % adjusted_count)

        score = 1 if num_points<50 else 3 if num_points>100 else 2
        IJ.log("Region " +str(i) + " score:  " + str(score))
        IJ.log("Region " + str(i) + " density: " + str(density_calc) + " cells/mm^2")
        
        if logging_bool:  # Logging saves output to location in plugin directory
            IJ.log("Log ID: " + dt_string + '\n')
            log_img = FileSaver(cropped_imp)
            log_img.saveAsTiff(plugin_wd + "logs/automatic_contrast_" + str(i) + '_' + dt_string + ".tiff")
            log_orig = FileSaver(no_contrast)
            log_orig.saveAsTiff(plugin_wd + "logs/automatic_unfiltered_" + str(i) + '_' + dt_string + ".tiff")
            
            with open(plugin_wd + "logs/automatic_region_" + str(i) + '_' + dt_string + ".csv", "w") as csvfile:
                log_writer = csv.writer(csvfile, delimiter=',', dialect="excel")
                for row in ROI_output:
                    log_writer.writerow(row)
        
        i += 1  # Increment counter

    avg_count = sum(all_counts)/float(len(all_counts))
    
    if len(all_counts) > 1:
        avg_score = 1 if avg_count <50 else 3 if avg_count >100 else 2  # Thresholds for cell count
        IJ.log("Score across " + str(len(all_counts)) +" regions: " + str(avg_score))
        IJ.log("Avg count across " + str(len(all_counts)) +" regions: %.2f" % avg_count)
        sum_differences_sq = 0
        for value in all_counts:
            sum_differences_sq += (value - avg_count)**2
        stdev_val = math.sqrt(sum_differences_sq / (len(all_counts)-1.0))
        IJ.log("Standard dev.: %.2f \n" % stdev_val)
    	
    # Deleting tile images saved in temp folder
    IJ.showStatus("Cleaning up temporary files")
    temp_files = glob.glob(temp_folder + '*')
    for f in temp_files:
        os.remove(f)

    IJ.setTool("point tool")

if __name__ in ['__builtin__', '__main__']:
    run_script()
