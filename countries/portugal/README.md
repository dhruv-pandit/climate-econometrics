# Portugal Climate Econometrics

Welcome to the Portugal section of the Climate Econometrics repository. This section contains datasets and notebooks specifically for Portugal that were used for (). Below, you'll find the structure and contents of this subfolder. 

## Folder Structure

```
countries/portugal/
│
├── datasets/
│   ├── counties/
│   └── era5/
│       ├── net_cdf_hourly/
│       │   └── *.nc (NetCDF files containing ERA5 complete hourly data)
│       └── *.xlsx (Excel datasets containing ERA5 data)
│       └── *.nc (NetCDF datasets containing ERA5 data)
│
└── notebooks/
    └── portugal_era5_processing.ipynb
    └── merge_era5_processing.ipynb
```

### Datasets

1. **ERA5**: Contains data from the ERA5 reanalysis project.
   - **Net_Cdf_hourly**: A subfolder containing NetCDF files with complete hourly data from ERA5 from 1990 to 2022. Monthly means are found in the parent folder from 1940-1980. 
   - **Excel Files**: Various Excel datasets (.xlsx) containing processed ERA5 data. Files with the suffix `_q` are quarterly data, while those with the suffix `_m` are monthly. 

### Notebooks

- **portugal_era5_processing.ipynb**: This Jupyter notebook processes the hourly ERA5 data into aggregated data. 
- **merge_era5_processing.ipynb**: This notebook merges and weights the data to a municipality level. 

