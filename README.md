<img
  src='https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png'
  height='48'
/>

# carbonplan / forest-emissions-tracking

**working repo for forest-emissions-tracking project**

[![GitHub][github-badge]][github]
![Build Status][]
![MIT License][]

[github]: https://github.com/carbonplan/forest-emissions-tracking
[github-badge]: https://flat.badgen.net/badge/-/github?icon=github&label
[build status]: https://flat.badgen.net/github/checks/carbonplan/forest-emissions-tracking
[mit license]: https://flat.badgen.net/badge/license/MIT/blue

[![Binder](https://mybinder.org/badge_logo.svg)](https://binder.pangeo.io/v2/gh/carbonplan/forest-emissions-tracking/master)

This repository includes example Jupyter notebooks and other utilities for a collaborative project CarbonPlan is working on involving tracking emissions related to deforestation and other land use changes.

This project is just getting going. Nothing here is final or complete.

We are in the process of incorporating the scripts and notebooks which were used to implement version 0 of the dataset. Some tips for reproducing this effort:

* the `scripts/aggregate_emissions.v0.py` script can be run to reproduce both the 3 km global emissions raster dataset and the country-average estimates. As a warning, depending on the size of the machine you're running on, you might encounter memory issues when dealing with the 30m datasets. For that reason, we opted to process the 30m tiles in serial. If you are struggling, you might want to check to ensure that your cluster isn't getting overloaded.


## license

All the code in this repository is MIT licensed.

## about us

CarbonPlan is a non-profit organization working on the science and data of carbon removal. We aim to improve the transparency and scientific integrity of carbon removal and climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/forest-emissions-tracking/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
