#' Access a feather store like a data frame
#'
#' These functions permit using a feather dataset much like a regular
#' (read-only) data frame without reading everything into R.
#'
#' They work by using [arrow::read_feather()] to read the data in as an Arrow
#' Table, an efficient data structure that supports many data-frame methods.
#' See the [Arrow package documentation](https://arrow.apache.org/docs/r/)
#' for more information.
#'
#' @inheritParams read_feather
#' @return An [arrow::Table]
#' @export
feather <- function(path) arrow::read_feather(path, as_data_frame = FALSE)

#' @rdname feather
#' @export
feather_metadata <- feather
