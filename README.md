# Multi-Comparison Matrix (MCM)

### This repository contains the software for our paper titled "An Approach to Multiple Comparison Benchmark Evaluations that is Stable Under Manipulation of the Comparate Set". This work has been done by [Ali Ismail-Fawaz](https://hadifawaz1999.github.io/), [Angus Dempster](https://dblp.uni-trier.de/pid/251/8985.html), [Chang Wei Tan](https://changweitan.com/), [Matthieu Herrmann](https://orcid.org/0000-0002-0074-470X), [Lynn Miller](https://au.linkedin.com/in/lynn-miller-bb1aa539), [Daniel Schmidt](https://research.monash.edu/en/persons/daniel-schmidt), [Stefano Berretti](http://www.micc.unifi.it/berretti/), [Jonathan Weber](https://www.jonathan-weber.eu/), [Maxime Devanne](https://maxime-devanne.com/), [Germain Forestier](https://germain-forestier.info/) and [Geoff I. Webb](https://i.giwebb.com/).

This work is done by ```list_of_authors```.

## Summary

This repo is a long term used benchmark method that generates a Multi-Comparison Matrix where the user ca choose whether to include a full pairwise multi-comparate comparison or to choose which ones to be included or excluded in the rows and columns of the matrix.

## Input Format

The input format is in a ```.csv``` file containing the statistics of each classifiers as the format of [this example](https://github.com/MSD-IRIMAS/Multi_Pairwise_Comparison/blob/main/results_example.csv).

## Usage of Code - Plot the MCM

In order for the user to plot the MCM, first thing is to load the ```.csv``` file into a ```pandas``` dataframe and feed it to the ```compare``` function. The user should specify the ```fig_savename``` parameter in order to save the output figure in ```pdf``` and ```png``` formats.

## Examples

Generating the MCM on the [following example](https://github.com/MSD-IRIMAS/Multi_Pairwise_Comparison/blob/main/results_example.csv) produces the following. To generate the following figure, the user follows this simple code:

```
import pandas as pd
from MCM import MCM

df_results = pd.read_csv('path/to/csv')

output_dir = '/output/directory/desired'

MCM.compare(
        output_dir=output_dir,
        df_results=df_results,
        fig_savename='heatmap',
        load_analysis=False
    )
```

<p align="center" width="100%">
<img src="heatmap.png" alt="heatmap-example"/>
</p>

Generating the MCM on the [following example](https://github.com/MSD-IRIMAS/Multi_Pairwise_Comparison/blob/main/results_example.csv) by excluding ```clf1``` and ```clf3``` from the columns.

```
import pandas as pd
from MCM import MCM

df_results = pd.read_csv('path/to/csv')

output_dir = '/output/directory/desired'

MCM.compare(
        output_dir=output_dir,
        df_results=df_results,
        excluded_col_comparates=['clf1','clf3'],
        fig_savename='heatline_vertical',
        load_analysis=False
    )
```

<p align="center" width="100%">
<img src="heatline_vertical.png" alt="heatline-vertical-example"/>
</p>

and by excluding them in the rows.

```
import pandas as pd
from MCM import MCM

df_results = pd.read_csv('path/to/csv')

output_dir = '/output/directory/desired'

MCM.compare(
        output_dir=output_dir,
        df_results=df_results,
        excluded_row_comparates=['clf1','clf3'],
        fig_savename='heatline_vertical',
        load_analysis=False
    )
```

<p align="center" width="100%">
<img src="heatline_horizontal.png" alt="heatline-horizontal-example"/>
</p>

## Requirements

The following python packages are required for the usage of the module:

1. ```numpy==1.23.5```
2. ```pandas==1.5.2```
3. ```matplotlib==3.6.2```
4. ```scipy==1.10.0```
5. ```baycomp==1.0```
