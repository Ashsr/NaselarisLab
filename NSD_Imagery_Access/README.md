# NSD Imagery Access

This is also a derived class from [NSD Access](https://github.com/tknapen/nsd_access) tool like [NSD Synthetic Access](https://github.com/Ashsr/NaselarisLab/tree/main/NSD_Synthetic_Access) \
Please install the NSD Access tool to use NSD Imagery Access. \
This is tailored to read in and analyse NSD Imagery data. \
It has similar functionality as NSD Access and has an additional function `get_expinfo` that returns a pandas dataframe with Stimulus information (Task type, Stimulus type, images shown/imagined, etc). \
The behavioral responses and the stimulus information are saved out as a csv file under the `meta_data` directory. \
There is also a [sample Jupyter Notebook](./Tryout_local_NSDIAccess.ipynb) that demonstrates the usage.
