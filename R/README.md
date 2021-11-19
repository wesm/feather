## Feather for R

Feather is file format designed for efficient on-disk serialisation of data frames that can be shared across programming languages (e.g. Python and R).

```R
library(feather)
write_feather(mtcars, "mtcars.feather")
mtcars2 <- read_feather("mtcars.feather")
```

Feather developement has continued in [Apache Arrow](https://arrow.apache.org/), and `feather` is now a wrapper around the [`arrow`](https://arrow.apache.org/docs/r/) package. We encourage you to update your workflows to use `arrow::write_feather()` and `arrow::read_feather()` directly.

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
