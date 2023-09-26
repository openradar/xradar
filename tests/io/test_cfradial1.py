import tempfile

import xarray as xr
from open_radar_data import DATASETS

import xradar as xd


def test_compare_sweeps():
    # Fetch the radar data file
    filename = DATASETS.fetch("cfrad.20080604_002217_000_SPOL_v36_SUR.nc")

    # Open the data tree
    dtree = xd.io.open_cfradial1_datatree(filename)

    # Create a temporary file to store the modified data tree
    with tempfile.NamedTemporaryFile(mode="w+b") as temp_file:
        # Save the modified data tree to the temporary file
        xd.io.to_cfradial1(dtree.copy(), temp_file.name, calibs=True)

        # Open the modified data tree
        dtree1 = xd.io.open_cfradial1_datatree(temp_file.name)

        # Iterate through sweep keys and compare DataArrays
        for grp in dtree1.groups:
            if "sweep" in grp:
                dtree1[grp]

        # Compare the values of the DataArrays for all sweeps
        for sweep_num in range(9):  # Assuming there are 9 sweeps
            xr.testing.assert_equal(
                dtree[f"sweep_{sweep_num}"].ds, dtree1[f"sweep_{sweep_num}"].ds
            )
