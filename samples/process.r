# Sample R Code
# =============
#
# This code demonstrates calling Alphan's Nuclei Detection web service using
# the R programming language.
#
# Note that you'll need to call this `install.packages` function one time to
# get the `httr` package which lets us make HTTP transactions. It only needs
# to be installed once. You can comment out this line for subsequent runs.

install.packages('httr', repos = 'https://cran.us.r-project.org')


# Next we make the `httr` package available to the rest of our code.

library(httr)


# Here we set up some convenient variables. Substitute your own credentials for
# `username` and `password`. For this example, we'll be working with the image
# file named `86_1.png`. Replace this with your own image you want to process.

username  <- 'myusername'
password  <- 's3cr3tpa55w0rd'
imagefile <- '86_1.png'
baseurl   <- 'https://edrn-labcas.jpl.nasa.gov/mlserve'


# Now we start the model by making an HTTP POST to the server. Here we set the
# name of the model to `unet_default` (which is the only model supported). We
# set the `is_extract_regionprops` flag to `True` which means we also want the
# region properties CSV file to be generated. Finally, we ask for a window size
# of 128; other values are 64 or 256.

starturl <- paste(
    baseurl,
    '/alphan/predict?model_name=unet_default&is_extract_regionprops=True&window=128',
    sep=''
)
response <- POST(
    starturl,
    accept_json(),
    authenticate(username, password),
    body = list(input_image = upload_file(imagefile, 'image/png')),
    encode = 'multipart'
)


# Now we need to let the model take some time to process the image and find
# nuclei. One minute should be enough.

Sys.sleep(60)


# Now we can ask for the completed processing. The POST above returned a small
# JSON dictionary with the URL suffix to add to the `baseurl` to retrieve the
# completed results.

resultsurl <- paste(baseurl, content(response)$`get results at`, sep='')
response <- GET(resultsurl, authenticate(username, password))


# Finally, we write the results to an ZIP file called `output.zip`. You can then
# use your system's built-in utilities to unzip the file, or use R's `unzip`
# function.

outputfile <- file('output.zip', 'wb')
writeBin(content(response), outputfile)
