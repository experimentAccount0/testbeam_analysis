# pyTestbeamAnalysis
A _very_ simple analysis of pixel-sensor data from testbeams. It is highly simplified at the cost of performance. Nevertheless all steps of a full analysis are included in one file in only 1000 lines of Python code.
Thus if you want to understand the basics of telescope reconstruction this code might help. If you want to have something fancy with good performance use e.g. " EUTelescope":http://eutelescope.web.cern.ch/.

# Installation
```bash
python setup.py install
```

Uninstall:
```bash
pip uninstall pyTestbeamAnalysis
```

# Example usage 
# Clusterizer
from clusterizer_example.py:
```python
from pyHitClusterizer.data_clusterizer import PyDataClusterizer
from pyHitClusterizer import data_struct

if __name__ == "__main__":
# create some fake data
    hits = np.ones(shape=(10, ), dtype=dtype_from_descr(data_struct.HitInfoTable))
    for i, hit in enumerate(hits):
        hit[0] = i / 2
        hit[1] = i / 2
        hit[2] = i + 2
        hit[3] = i % 2 + 10
        hit[4] = i % 3 + 1
    hits[8]['event_number'] = 3

    # create results arrays to be filled by the clusterizer
    cluster_hits = np.zeros_like(hits, dtype=dtype_from_descr(data_struct.ClusterHitInfoTable))
    cluster = np.zeros_like(hits, dtype=dtype_from_descr(data_struct.ClusterInfoTable))

    # create clusterizer object
    clusterizer = PyDataClusterizer()

    # all working settings are listed here, the std. values are used here
    clusterizer.set_debug_output(False)
    clusterizer.set_info_output(False)
    clusterizer.set_warning_output(True)
    clusterizer.set_error_output(True)

    clusterizer.create_cluster_info_array(True)
    clusterizer.create_cluster_hit_info_array(True)  # std. setting is False

    clusterizer.set_x_cluster_distance(1)  # cluster distance in columns
    clusterizer.set_y_cluster_distance(2)  # cluster distance in rows
    clusterizer.set_frame_cluster_distance(4)   # cluster distance in time frames

    # main functions
    clusterizer.set_cluster_hit_info_array(cluster_hits)  # tell the array to be filled
    clusterizer.set_cluster_info_array(cluster)  # tell the array to be filled
    clusterizer.add_hits(hits)  # cluster hits
```

## Full Analysis
Comming soon...



