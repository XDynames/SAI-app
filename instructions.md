Stoma AI (SAI) is a joint research outcome developed through inter-disciplinary collaboration between the ARC Center of Excellence for Plant Energy Biology and the Australian Institute for Machine Learning.
SAI aims to be an easy to use tool for plant physiologist to automate the arduous manual cell measurement tasks required by many experiments.

### Instructions
Start by selecting a mode in the sidebar
There are currently five options - Instructions, View Example Images, View Example Output, Measure An Image and Measure Your Data.

### View Example Images
Here you are able to look through a series of annotated microscope images used in our experiments.
Examples are divided into either barley or arabidopsis which can be switched between using the plant type drop down.
Individual images from each type of plant can then be selected.
By enabling either Show Human Measurements or Show Model Predictions annotations will be draw onto the image.
Each type of measurement can be toggled on or off in the side bar so you can better inspect each type without overlap.
A confidence threshold slider will become available when Show Model Predictions is enables.
Each prediction is assigned a score; representing how strong the response from the model is.

### View Example Output
Displays an example of how .csv files exported from the application will be formatted.

### Measure Your Own Samples
Currently this is disabled in the online demo version of the application due to computation limits of our web hosting.
If you want to use this application in your own research follow this [link](https://github.com/XDynames/SAI-app) to our github project and follow the local installation instructions.
#### Measure An Image
For a quick demonstration of SAI you can use the Measure An Image feature.
Set the plant type to barley or arabidopsis based on the content of the image you will provide.
Upload your slide image using drag and drop or the file browser.
By default the summary statistics are calculated using pixels, by entering a value for camera calibration these will update the summary to μm and μm².
#### Measure Your Data
This is the main component of the tool and allows automated measuring of large collections of microscopy images.
Use the file browser to select the folder which contains your microscopy images and hit run.
SAI is now measuring all those samples for you.
On completion you will be able to download a .csv file with SAI's measurements for each pore and save out visualisations of SAI's measurements overlaid on your samples (Like those seen in View Example Images).
Each pore measured by SAI is given a ID number which you can use to cross-check measurements in the .csv with those in the visualisations.

<div align="center">
<b>Links

| Publication TBA | [Tool](https://github.com/XDynames/SAI-app) | Deep Learning Code TBA |
|:---:|:---:|:---:|

</div>