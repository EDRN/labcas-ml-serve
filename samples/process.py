# Sample R Code
# =============
#
# This code demonstrates calling Alphan's Nuclei Detection web service using
# the Python programming language, version 3.
#
# This sample uses the Python "requests" library which you'll need to install
# first with a command sequence similar to the following:
#
#     python3 -m venv venv
#     source venv/bin/activate
#     pip install requests
#
# You can then execute this file with: `python process.py`

from requests.auth import HTTPBasicAuth
import requests, time


# Here we set up some convenient variables. Substitute your own credentials for
# `username` and `password`. For this example, we'll be working with the image
# file named `86_1.png`. Replace this with your own image you want to process.

username  = 'myusername'
password  = 's3cr3tpa55w0rd'
imagefile = '86_1.png'
baseurl   = 'https://edrn-labcas.jpl.nasa.gov/mlserve'


# Now we start the model by making an HTTP POST to the server. Here we set the
# name of the model to `unet_default` (which is the only model supported). We
# set the `is_extract_regionprops` flag to `True` which means we also want the
# region properties CSV file to be generated. Finally, we ask for a window size
# of 128; other values are 64 or 256.

starturl = baseurl + '/alphan/predict?model_name=unet_default&is_extract_regionprops=True&window=128'
files = {'input_image': (imagefile, open(imagefile, 'rb'), 'image/png')}
auth = HTTPBasicAuth(username, password)
response = requests.post(starturl, files=files, auth=auth)

# Now we need to let the model take some time to process the image and find
# nuclei. One minute should be enough.

time.sleep(60)

# Now we can ask for the completed processing. The POST above returned a small
# JSON dictionary with the URL suffix to add to the `baseurl` to retrieve the
# completed results.

resultsurl = baseurl + response.json()['get results at']
response = requests.get(resultsurl, auth=auth)

# Finally, we write the results to an ZIP file called `output.zip`. You can then
# use your system's built-in utilities to unzip the file, or use Python's
# `zipfile` module.

with open('output.zip', 'wb') as outputfile:
    outputfile.write(response.content)
