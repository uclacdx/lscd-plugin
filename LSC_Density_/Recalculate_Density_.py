from ij import IJ
from ij.io import FileSaver
from ConfigParser import SafeConfigParser
from datetime import datetime
import csv

def density_recalc():
    # Config setup
    plugin_wd = IJ.getDirectory('current') + 'scripts/'
    config = SafeConfigParser()
    config.read(plugin_wd + 'plugin_config.ini')
    logging_bool = config.getboolean('path_info','logging')
    
    dt_now = datetime.utcnow()
    dt_string = dt_now.strftime("%y%m%d%H%M%S")
   
    current = IJ.getImage()
    img_name = current.getShortTitle()
    overlay = current.getOverlay()
    current_size = overlay.size()
    density_calc = (1000 ** 2 * current_size) / 40000
    IJ.log("Updated " + img_name + " cell count: " + str(current_size))
    IJ.log("Updated " + img_name + " density: " + str(density_calc) + " cells/mm^2")
    if logging_bool:
        IJ.log("Log ID: " + dt_string + '\n')
        log_img = FileSaver(current)
        log_img.saveAsTiff(plugin_wd + "logs/manual_"+ img_name + '_' + dt_string + ".tiff")   

        iterator = overlay.iterator()
        with open(plugin_wd + "logs/manual_" + img_name + '_' + dt_string + ".csv", "w") as csvfile:
            log_writer = csv.writer(csvfile, delimiter=',', dialect="excel")
            while iterator.hasNext():
                write_out = str(iterator.next())[13:].strip('[]')
                x = int(write_out.split(',')[0])
                y = int(write_out.split(',')[1][3:])
                log_writer.writerow([x,y])

                        
            

if __name__ in ['__builtin__', '__main__']:
    density_recalc()
