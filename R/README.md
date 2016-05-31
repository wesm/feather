## Feather for R

Feather is file format designed for efficient on-disk serialisation of data frames that can be shared across programming languages (e.g. Python and R).

## Installation
###Binaries
Feather is now [on CRAN](https://cran.r-project.org/web/packages/feather/index.html), and binaries exist for both Windows and OS X Mavericks.

Install from CRAN with:
```R
# install.packages("feather")
```

###Source
Feather uses C++11, so if you're on Windows, and you want to install from source, you will first need to install the [most recent version of Rtools](https://cran.r-project.org/bin/windows/Rtools/), currently version 3.3.0.1959.

Install from CRAN with:
```R
# install.packages("feather", type = 'source')
```

Install from Github with:
```R
# install.packages("devtools")
devtools::install_github("wesm/feather/R")
```
