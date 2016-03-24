## Feather for R

Feather is file format designed for efficient on-disk serialisation of data frames that can be shared across programming languages (e.g. Python and R).

## Installation

Install from Github with:

```R
# install.packages("devtools")
devtools::install_github("wesm/feather/R")
```

Feather uses C++11, so if you're on windows, you'll require the [experimental gcc 4.93 toolchain](https://github.com/rwinlib/r-base/wiki/Testing-Packages-with-Experimental-R-Devel-Build-for-Windows). This should become official when R 3.3.0 is released, which is also when we'll submit it to CRAN, so installation will eventually painless.
