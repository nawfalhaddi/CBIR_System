# USAGE
# python index.py --dataset dataset --index index.csv

# import the necessary packages
from SearchEngine.colordescriptor import ColorDescriptor
from SearchEngine.texturedescriptor import TextureDescriptor
from SearchEngine.gabordescriptor import GaborDescriptor
import argparse
import glob
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "lien pour dataset")
ap.add_argument("-i", "--index", required = True,
	help = "lien pour index.csv")
args = vars(ap.parse_args())

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))
ht= TextureDescriptor()
gb= GaborDescriptor()

# open the output index file for writing
output = open(args["index"], "w")

# use glob to grab the image paths and loop over them
for imagePath in glob.glob(args["dataset"] + "/*.png"):
    # extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
    imageID = imagePath[imagePath.rfind("\\") + 1:]
    image = cv2.imread(imagePath)
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
    # describe the image
    features = list(np.array(cd.describe(image)))
    features = features+list(np.array(ht.extract_features(image)))
    """
    features = list(np.array(gb.extract_features(image)))
    """
    # write the features to file
    features = [str(f) for f in features]
    output.write("%s,%s\n" % (imageID, ",".join(features)))

# close the index file
output.close()