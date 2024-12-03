# Flood Inundation Mapping Predictions Evaluation Framework (FIMPEF)

<div align="left">
  <img src="https://github.com/user-attachments/assets/a6af2a80-f6e2-46e0-a17b-791303a3a705" alt="SDML" width="150" style="margin-right: 15px; float: left;">
  <p style="display: inline-block; vertical-align: top;">
    This repository provides a user-friendly Python package (version 0.1.2) and source code for the automatic evaluation of flood inundation maps.  
    It is developed by the Surface Dynamics Modeling Lab (SDML) in the Department of Geography and Environment at The University of Alabama, United States.
  </p>
</div>


# Background

The accuracy of the flood inundation mapping (FIM) is critical for model development and disaster preparedness. The evaluation of flood maps from different sources using geospatial platforms can be tedious and requires repeated processing and analysis for each map. These preprocessing steps include extracting the correct flood extent, assigning the same projection system to all the maps, categorizing the maps as binary flood maps, removal of permanent water bodies, etc. This manual data processing is cumbersome and prone to human error.

To address these issues, we developed Flood Inundation Mapping Prediction Evaluation Framework (FIMPEF), a Python-based FIM evaluation framework capable of automatically evaluating flood maps from different sources. FIMPEF takes the advantage of comparing multiple target datasets with large benchmark datasets. It includes an option to incorporate permanent waterbodies as non-flood pixels with a user input file or pre-set dataset. In addition to traditional evaluation metrics, it can also compare the number of buildings inundated using a user input file or a pre-set dataset.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7cbe7691-c680-43d5-99bf-6ac788670921" width="500" />
</p>


# Repository Structure
```bash
FIMPEF/
├── Input Rasters/
│   └── Case 1/ 
│       └── RG_benchmark.tif  (Benchmark FIM for Hurricane Mathew, Oct 09, 2016, North Carolina. Make sure to enter the name 'benchmark' while naming the raster)
│       └── OWP_09_NC.tif     (Model FIM for Hurricane Mathew, Oct 09, 2016, North Carolina. (NOAA OWP HAND FIM))
├── PWB/
│   └── PWB.shp               (Shapefile of Permanent Water Bodies.)
├── BuildingFootprint/
│   └── NC_bldg.shp            (Geopackage of building footprints.The building footprint used is Microsoft release under Open Data Commons Open Database Liocence. Here is the link https://automaticknowledge.co.uk/us-building-footprints/ User can download the building footprints of the desired states from this link.)
├── FIMPEFfunctions.py         (Contains all functions associated with the notebook)
├── FIMPEF.ipynb               (The main notebook code to get FIM)
├── FIMPEF_package.ipynb       (FIMPEF package version 0.1.2)
└── README.md                 (This file)
```
# Main Directory Structure
The main directory contains the main folder. Inside the main folder there are sub folders with the case studies. If a user has three case studies then user need to prepare three folders. Inside each folder there should be a B-FIM with a 'benchmark' name assigned in it and different M-FIM in tif format.
<div align="center">
  <img width="300" alt="image" src="https://github.com/user-attachments/assets/3329baf0-e5d4-4f54-a5a2-278c34b68ac8">
</div>

## Permanent Water Bodies
In this work the 'USA Detailed Water Bodies' from ARCGIS hub is used. Here is the link https://hub.arcgis.com/datasets/esri::usa-detailed-water-bodies/about. User can input their own permanent water bodies shapefile as .shp and .gpkg format.

## Building Footprints
The building footprint used is Microsoft release under Open Data Commons Open Database Licence. Here is the link https://automaticknowledge.co.uk/us-building-footprints/ User can download the building footprints of the desired states from this link.

# Usage

For directly using the package, the user can use the following code in a Jupyter notebook:

```bash
!pip install fimpef==0.1.2
```
For using the source code, simply run the FIMPEF.ipynb importing the FIMPEFfunctions.py.

# Outputs
The output from FIMPEF includes generated files in TIFF, SHP, CSV, and PNG formats, all stored within the "Case 1" folder. Users can visualize the TIFF files using any geospatial platform. The TIFF files consist of the binary Benchmark-FIM (Benchmark.tif), Model-FIM (Candidate.tif), and Agreement-FIM (Contingency.tif). The shp files contain the boundary of the generated flood extent. The png files include the Agreement map, Performance Metrics, and Building Footprint Statistics.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a1cfeb14-45b2-4c77-96d4-7ce3ae82c3e8" width="350" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/5e6bdd33-8b4c-4ec7-9bc9-18fcd1e39cdb" width="450" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e0348548-e380-422e-9b97-c0b859ac6ac7" width="500" />
</p>

## Citation

If you use this repository or its components in your work, please cite it as follows:

Dipsikha Devi, Supath Dhital, Dinuke Munasinghe, Anupal Baruah, Sagy Cohen. "FIMPEF" GitHub, 2024, .https://github.com/dipsikha-devi/FIMPEF
### **Acknowledgements**
| | |
| --- | --- |
| ![alt text](https://ciroh.ua.edu/wp-content/uploads/2022/08/CIROHLogo_200x200.png) | Funding for this project was provided by the National Oceanic & Atmospheric Administration (NOAA), awarded to the Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement with The University of Alabama.

### **For More Information**
Contact <a href="https://geography.ua.edu/people/sagy-cohen/" target="_blank">Dr. Sagy Cohen</a>
 (sagy.cohen@ua.edu)
Dr. Dipsikha Devi, (ddevi@ua.edu)
Supath Dhittal,(sdhital@crimson.ua.edu)
