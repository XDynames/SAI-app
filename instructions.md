*Add AIML/UofA/Agri logo

*Add Authorship, Paper, Code Links ect

Start by selecting a mode in the sidebar
There are currently four options - Instructions, View Example Images, View Example Output, Upload An Image.

## View Example Images
Here you are able to look through a series of annotated cell biopsy slides used in our experiments.
Examples are divided into either barley or arabidopsis which can be switched between using the plant type drop down.
Individual images from each type of plant can then be selected.
By enabling either Show Human Measurements or Show Model Predictions annotations will be draw onto the image.
Each type of measurement can be toggled on or off in the side bar so you can better inspect each type without overlap.
A confidence threshold slider will become available when Show Model Predictions is enables.
Each prediction is assigned a score; representing how strong the response from the model is.

## View Example Output
Displays an example of how .csv files exported from the application will be formatted.

## Upload An Image
This section provides an opportunity for your to interact with our model by uploading a file of your own.
If required set the plant type to barley or arabidopsis based on the content of the image you will provide.
Upload your slide image by either dragging and drop or browsing files.
By default the summary statistics are calculated using pixels as the default unit.
Entering your own values for camera calibration and image area will update the summary with SI units.