<img
  src='https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png'
  height='48'
/>

# carbonplan / trace

**working repo for carbonplan's climate trace project**

[![GitHub][github-badge]][github]
[![Build Status]][actions]
![MIT License][]
[![Binder](https://mybinder.org/badge_logo.svg)](https://binder.pangeo.io/v2/gh/carbonplan/trace/main)

[github]: https://github.com/carbonplan/trace
[github-badge]: https://badgen.net/badge/-/github?icon=github&label
[build status]: https://github.com/carbonplan/trace/actions/workflows/main.yaml/badge.svg
[actions]: https://github.com/carbonplan/trace/actions/workflows/main.yaml
[mit license]: https://badgen.net/badge/license/MIT/blue

This repository includes example Jupyter notebooks and other utilities for a collaborative project CarbonPlan is working on involving tracking emissions related to biomass losses from forests.

This project is a work in progress. Nothing here is final or complete.

We have completed the scripts and notebooks for delivery of version 0 of the dataset (`carbonplan_trace/v0`) which largely reproduces and extends work by Zarin et al (2016) as hosted on the Global Forest Watch platform. 

Input datasets include:
* aboveground biomass for year 2000 (Hansen et al (2020))
* binary masks of tree cover loss year for 2001-2020 (Hansen et al (2020))
* country boundary shapefile from the Database of Global Administrative Areas (GADM) version 3.6. *Note: Geopolitical boundaries that have changed over the period of record will be tagged to the static country designation as defined in GADM v3.6.* 

Some tips for reproducing this effort:

* The `scripts/aggregate_emissions.v0.py` script can be run to reproduce both the 3 km global emissions raster dataset and the country-average estimates. As a warning, depending on the size of the machine you're running on, you might encounter memory issues when dealing with the 30m datasets. For that reason, we opted to process the 30m tiles in serial. If you are struggling, you might want to check to ensure that your cluster isn't getting overloaded.
* We recommend using the sample notebook `notebooks/blogpost_sample_notebook.ipynb` as a starting point to introduce yourself to the structure of the data and how to work with a high resolution global product. We emphasize that the 3km product may be sufficient for some users.

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/) licensed. When possible, the data used by this project is licensed using the [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/) license. We include attribution and additional license information for third party datasets, and we request that you also maintain that attribution if using this data.

## about us

CarbonPlan is a non-profit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of carbon removal and climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/trace/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
