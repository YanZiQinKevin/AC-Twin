# Data and model availability

The release inventory is [release/manifest.csv](release/manifest.csv). It records original relative paths, file sizes, SHA-256 checksums, product categories and storage routes. The inventory distinguishes files supplied in this checkout from larger files requiring external download.

## Included in Git

- Model checkpoints and weights below 10 MiB per file, including the Merge and MetaDrive model stacks and supervised feature controls.
- Smaller generated highway-env datasets, collection metadata and split information.
- Frozen experiment reports, per-group evaluation products below the threshold, negative results and terminal decisions.
- Current figure/table source data, editable exports and rendered evaluation illustrations.

Historical unsuccessful experiments remain labelled by their own reports. Their presence does not change the confirmed claims or authorize reuse of a consumed confirmation set.

## Large generated products

Files at least 10 MiB are omitted from Git. Their original paths, sizes and checksums appear in [release/downloads.json](release/downloads.json). Restore each downloaded file to its recorded relative path.

A null URL and `availability: awaiting_host` mean that a download location has not yet been supplied. Such a record is an inventory entry, not a claim of online availability. The release summary reports included and externally stored byte counts separately.

## Third-party sources and training-data exclusion

- [Highway-env](https://github.com/Farama-Foundation/HighwayEnv) generates the simulated Merge, intersection and roundabout data.
- Obtain [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/) scenario release v1.2.0, `training_20s`, from the provider, subject to its access terms.
- Conversion and simulation use [ScenarioNet](https://github.com/metadriverse/scenarionet) and [MetaDrive](https://github.com/metadriverse/metadrive).

Raw Waymo TFRecords and converted third-party scenario databases are not redistributed. MetaDrive role-`train` sample arrays and training prediction caches are excluded. Trained weights, training diagnostics and validation/screen/confirmation evaluation products are separate categories and are included or indexed according to size. The dataset split names used within this study are distinct from Waymo's provider-level `training_20s` split name.

[release/excluded_files.json](release/excluded_files.json) records the actual exclusion decisions within the inspected workspace. Third-party databases outside that workspace are never traversed or copied.

No dataset DOI, licence grant or unrestricted third-party redistribution right is implied by this inventory.
