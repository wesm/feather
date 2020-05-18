#' Access a feather store like a data frame
#'
#' These functions permit using a feather dataset much like a regular
#' (read-only) data frame.
#'
#' @inheritParams read_feather
#' @return An object of class \code{feather}
#' @export
feather <- function(path) {
  arrow::read_feather(path, as_data_frame = FALSE)
}
