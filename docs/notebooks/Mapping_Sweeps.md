---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
  main_language: python
kernelspec:
  display_name: Python 3
  name: python3
---

# Map Over Sweeps

## map_over_sweeps Accessor and Decorator

+++

Have you ever wondered how to efficiently apply operations to a full volume of radar sweeps, rather than processing each sweep individually?

Xradar has the solution: the `map_over_sweeps` accessor. In this notebook, we’ll explore how you can leverage Xradar’s powerful `map_over_sweeps` functionality to perform volume-level operations on radar data.

In radar data analysis, it's common to work with multiple sweeps in a radar volume. Xradar allows you to apply custom functions across the entire dataset with ease, making complex operations, such as filtering reflectivity or calculating rain rate, efficient and scalable.

Here's what you'll learn in this notebook:
- How to load and inspect radar data using Xradar.
- How to apply functions to process all radar sweeps in one go using both conventional and decorator-based methods.
- How to visualize radar variables like reflectivity and rain rate before and after processing.

Let’s get into it!

+++

## Imports

```{code-cell}
import matplotlib.pyplot as plt
from matplotlib import gridspec
from open_radar_data import DATASETS

import xradar as xd
```

## Load Read/Data

```{code-cell}
# Fetch the sample radar file
filename = DATASETS.fetch("sample_sgp_data.nc")

# Open the radar file into a DataTree object
dtree = xd.io.open_cfradial1_datatree(filename)
dtree = dtree.xradar.georeference()
```

## Exploring Data Variables

```{code-cell}
display(dtree)
```

```{code-cell}
for var in dtree["sweep_0"].ds:
    print(var)
```

## Map Over Sweeps

+++

We define custom functions like `filter_radar` that filters radar reflectivity based on certain conditions, and `calculate_rain_rate` which is self explanatory, and use `map_over_sweeps` accessor to implement these.

+++

## Example #1

```{code-cell}
# It is just a demonstration, you can ignore the logic
def filter_radar(ds):
    ds = ds.assign(
        DBZH_Filtered=ds.where(
            (ds["corrected_reflectivity_horizontal"] > 10)
            & (ds["corrected_reflectivity_horizontal"] < 70)
            & (ds["copol_coeff"] > 0.85)
        )["corrected_reflectivity_horizontal"]
    )
    return ds
```

```{code-cell}
# Apply the function across all sweeps
dtree = dtree.xradar.map_over_sweeps(filter_radar)
```

## Comparison

+++

Now, let's compare the unfiltered and filtered reflectivity

```{code-cell}
# Set a larger figure size with a wider aspect ratio for readability
fig, ax = plt.subplots(figsize=(10, 4))

# Plot the unfiltered and filtered reflectivity with clear distinctions
ax.plot(
    dtree["sweep_0"]["range"],
    dtree["sweep_0"]["corrected_reflectivity_horizontal"].sel(
        azimuth=100, method="nearest"
    ),
    alpha=0.7,
    lw=2,
    linestyle="-",
    color="m",
    label="Unfiltered",
)

ax.plot(
    dtree["sweep_0"]["range"],
    dtree["sweep_0"]["DBZH_Filtered"].sel(azimuth=100, method="nearest"),
    alpha=0.7,
    lw=2,
    linestyle="--",
    color="green",
    label="Filtered",
)

ax.set_xlim(1000, 40_000)
ax.set_ylim(0, 50)

# Set title and labels with enhanced font sizes
ax.set_title(
    "Compare Unfiltered and \
Filtered Reflectivity Variables (Azimuth = 100°)",
    fontsize=16,
    pad=15,
)
ax.set_ylabel("Reflectivity [dBZ]", fontsize=14)
ax.set_xlabel("Range [m]", fontsize=14)

# Add minor ticks and a grid for both major and minor ticks
ax.minorticks_on()
ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# Adjust legend to avoid overlapping with the plot, and make it clear
ax.legend(loc="upper right", fontsize=12, frameon=True)

# Apply a tight layout to avoid label/title overlap
plt.tight_layout()

# Show the plot
plt.show()
```

## Example #2

```{code-cell}
# Define a function to calculate rain rate from reflectivity
def calculate_rain_rate(ds, ref_field="DBZH"):
    def _rain_rate(dbz, a=200.0, b=1.6):
        Z = 10.0 ** (dbz / 10.0)
        return (Z / a) ** (1.0 / b)

    ds = ds.assign(RAIN_RATE=_rain_rate(ds[ref_field]))
    ds["RAIN_RATE"].attrs = {"units": "mm/h", "long_name": "Rain Rate"}
    return ds
```

```{code-cell}
# Apply the function across all sweeps
dtree = dtree.xradar.map_over_sweeps(calculate_rain_rate, ref_field="DBZH_Filtered")
dtree
```

If you expand the `dtree` groups and sweeps, you'll notice that the `RAIN_RATE` variable has been successfully added to the DataTree. This confirms that the function was applied across all sweeps, incorporating the calculated rain rate into the dataset.

```{code-cell}
# Create a figure with a GridSpec layout
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.1)  # Reduced spacing between plots

# Define common plotting settings
vmin = 0
vmax = 50
cmap = "viridis"

# Create the subplots without shrinking due to colorbar
axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

# Plot each sweep on the respective subplot
for i, ax in enumerate(axes):
    sweep = f"sweep_{i}"
    im = dtree[sweep]["RAIN_RATE"].plot(
        x="x",
        y="y",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ax=ax,
        add_colorbar=False,  # Disable individual colorbars
    )
    ax.set_title(f"Rain Rate for {sweep.replace('_', ' ').capitalize()}", fontsize=14)

    # Turn off ticks and labels for inner subplots
    if i in [0, 1]:  # Top row
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if i in [1, 3]:  # Right column
        ax.set_yticklabels([])
        ax.set_ylabel("")

# Create a single shared colorbar outside the subplots
# Adjust [left, bottom, width, height] for colorbar position
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
fig.colorbar(im, cax=cbar_ax, label="Rain Rate (mm/hr)")

# Set a main title for the entire figure
fig.suptitle("Rain Rate Across Different Sweeps", fontsize=16, fontweight="bold")

# Show the plot
plt.show()
```

## With decorator

```{code-cell}
@xd.map_over_sweeps
def calculate_rain_rate2(ds, ref_field="DBZH"):
    def _rain_rate(dbz, a=200.0, b=1.6):
        Z = 10.0 ** (dbz / 10.0)
        return (Z / a) ** (1.0 / b)

    ds = ds.assign(RAIN_RATE2=_rain_rate(ds[ref_field]))
    ds.RAIN_RATE2.attrs = {"units": "mm/h", "long_name": "Rain Rate"}
    return ds


# invocation via decorator + pipe
dtree3 = dtree.pipe(calculate_rain_rate2, ref_field="DBZH_Filtered")
display(dtree3["sweep_0"])
```

## Conclusion

+++

In this notebook, we demonstrated how to handle and process radar data efficiently using Xradar’s `map_over_sweeps` accessor. By applying operations across an entire volume of radar sweeps, you can streamline your workflow and avoid the need to manually process each sweep.

We explored two key use cases:
- **Filtering Reflectivity**: We applied a custom filtering function across all sweeps in the radar dataset, allowing us to isolate meaningful reflectivity values based on specific criteria.
-  **Calculating Rain Rate**: Using the radar reflectivity data, we calculated the rain rate for each sweep, demonstrating how to perform scientific computations across multiple sweeps with minimal effort.

The `map_over_sweeps` functionality in Xradar opens the door to performing various radar data processing tasks efficiently. Whether it's filtering, calculating derived quantities like rain rate, or applying more complex algorithms, Xradar simplifies working with radar volumes, making it easier to scale your analysis.
