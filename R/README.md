## Feather for R

[![Build Status](https://travis-ci.org/wesm/feather.svg?branch=master)](https://travis-ci.org/wesm/feather)

Feather is file format designed for efficient on-disk serialisation of data frames that can be shared across programming languages (e.g. Python and R).

```R
library(feather)
write_feather(mtcars, "mtcars.feather")
mtcars2 <- read_feather("mtcars.feather")
```

## Installation

Install the released version from CRAN:

```R
# install.packages("feather")
```

Or the development version from GitHub:

```R
# install.packages("devtools")
devtools::install_github("wesm/feather/R")
```
