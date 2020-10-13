# import needed packages
from requests import exceptions
import argparse
import requests
import cv2
import os

import keys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
    help="search query to search Bing Image API")
ap.add_argument("-o", "--output", required=True,
    help="path to putput directory of images")
args = vars(ap.parse_args())

# set API key along with the max number of results per search
# and the group size for results ( max 50 per request)
API_KEY = keys.BING_API_KEY
MAX_RESULTS = 250
GROUP_SIZE = 50

# set the endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

# list of exceptions we might see when trying to download images
EXCEPTIONS = set([IOError, FileNotFoundError, exceptions.RequestException,
    exceptions.HTTPError, exceptions.ConnectionError, exceptions.Timeout])


