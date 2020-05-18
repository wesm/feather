#' Read and write feather files.
#'
#' @param path Path to feather file
#' @param columns Columns to read (names or indexes). Default: Read all columns.
#' @param version integer in `c(1, 2)` indicating the Feather format version to
#'   write. For compatibility, the default for `feather::write_feather()` is
#'   `1`.
#' @param ... Additional arguments passed to the `arrow::` functions
#' @return Both functions return a tibble/data frame. \code{write_feather}
#'   invisibly returns \code{x} (so you can use this function in a pipeline).
#' @examples
#' mtcars2 <- read_feather(feather_example("mtcars.feather"))
#' mtcars2
#' @export
#' @importFrom arrow read_feather write_feather
#' @importFrom rlang !! enquo
read_feather <- function(path, columns = NULL, ...) {
  arrow::read_feather(path, col_select = !!enquo(columns), ...)
}

#' @rdname read_feather
#' @param x A data frame to write to disk
#' @export
write_feather <- function(x, path, version = 1, ...) {
  arrow::write_feather(x, path, version = version, ...)
}

#' Retrieve metadata about a feather file
#'
#' Returns the dimensions, field names, and types; and optional dataset
#' description.
#'
#' @param path Path to feather file
#' @return A list with class "feather_metadata".
#' @export
feather_metadata <- feather
