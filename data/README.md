# Data

No data is included in this repository. Get each dataset from its provider and accept its terms.

| Dataset | Source | Licence |
|---|---|---|
| CERT Insider Threat Test Dataset r4.2 and r5.2 | https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247 | See provider |
| SPEDIA v1.0 | https://doi.org/10.5281/zenodo.15495572 (collection tools: https://github.com/UPM-RSTI/SPEDIA-Dataset) | CC BY 4.0 |
| LANL Comprehensive Multi-Source Cyber-Security Events | https://doi.org/10.17021/1179829 | CC0 |

## Expected layout

The code reads the user-day tables from `$FITD_ROOT/data/`:

```
data/
├── cert42_user_day_dev.parquet
├── cert52_user_day_dev.parquet
├── spedia_user_day_dev.parquet
├── lanl_user_day_dev.parquet
└── derived/            # LANL rebuilds (E-10) and other derived tables
```

How to build these tables, and their SHA-256 hashes, are given in the main [README](../README.md#building-the-user-day-tables).
