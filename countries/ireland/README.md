# Ireland Climate Econometrics

Welcome to the Ireland section of the Climate Econometrics repository. This section contains datasets and notebooks specifically for Ireland. Below, you'll find the structure and contents of this subfolder.

## Folder Structure

```
countries/ireland/
│
├── datasets/
│   ├── counties/
│   └── era5/
│       ├── all_data/
│       │   └── *.nc (NetCDF files containing ERA5 complete hourly data)
│       └── *.xlsx (Excel datasets containing ERA5 data)
│
└── notebooks/
    └── ireland_era5_processing.ipynb
```

### Datasets

1. **county_data**: Contains shapefiles specific to counties in Ireland.
2. **ERA5**: Contains data from the ERA5 reanalysis project.
   - **AllData**: A subfolder containing NetCDF files with complete hourly data from ERA5 from 1990 to 2022. Monthly means are found in the parent folder from 1940-1980. 
   - **Excel Files**: Various Excel datasets (.xlsx) containing processed ERA5 data. Files with the suffix `_q` are quarterly data, while those with the suffix `_m_` are monthly. 

### Notebooks

- **ireland_era5_processing.ipynb**: This Jupyter notebook processes the hourly ERA5 data into county-aggregated data. 

