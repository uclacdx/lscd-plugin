//@AutoInstall

var size = 272;
var leftButton = 16;
var alt = 8;

macro "Square Tool - C037R0055Rd355R7d55" {
	moving = false;
	getCursorLoc(x, y, z, flags);
	index = Overlay.indexAt(x,y);	
	if (index>=0 && flags&alt!=0) {  // delete?
		Overlay.removeSelection(index);
		exit;
	}
	if (index>=0) {  // move
		Overlay.activateSelection(index);
		moving = true;
	}
	while (flags&leftButton!=0) {
		if (moving)
			Overlay.moveSelection(index, x-size/2, y-size/2);
		else
			makeRectangle(x-size/2, y-size/2, size, size);
		wait(25);
		getCursorLoc(x, y, z, flags);
	}
	if (!moving)
		if (selectionType() == -1)
			IJ.redirectErrorMessages();
		else {
			Overlay.addSelection;
			IJ.redirectErrorMessages();
		}
	Roi.remove;
}

macro "Square Tool Options" {
       size = getNumber("Size (default 272): ", size);
    }