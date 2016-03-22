#' Get path to feather example files
#'
#' @export
#' @keywords internal
feather_example <- function(x) {
  system.file("feather", x, mustWork = TRUE, package = "feather")
}

dim_desc <- function(x) {
  paste0("[", paste0(big_mark(x), collapse = " x "), "]")
}

big_mark <- function(x, ...) {
  mark <- if (identical(getOption("OutDec"), ",")) "." else ","
  formatC(x, big.mark = mark, ...)
}
