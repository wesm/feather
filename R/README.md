## Feather for R

Feather is file format designed for efficient on-disk serialisation of data frames that can be shared across programming languages (e.g. Python and R).

## Installation

Install from Github with:

```R
# install.packages("devtools")
devtools::install_github("wesm/feather/R")
```

Feather uses C++11, so if you're on Windows, you'll first need to install the [experimental gcc 4.93 toolchain](https://github.com/rwinlib/r-base/wiki/Testing-Packages-with-Experimental-R-Devel-Build-for-Windows). 

Installation will eventually become painless. When R 3.3.0 is released, not only should this toolchain become the official toolchain for building extensions under Windows but we'll also be submitting Feather to CRAN.
