#' @useDynLib feather
#' @importFrom Rcpp sourceCpp
NULL

#' Read and write feather files.
#'
#' @param path Path to feather file
#' @name feather
#' @return Both functions return a tibble/data frame. \code{write_feather}
#'   invisibly returns \code{x} (so you can use this function in a pipeline).
NULL

#' @rdname feather
#' @export
read_feather <- function(path) {
  readFeather(path)
}

#' @rdname feather
#' @export
write_feather <- function(x, path) {
  writeFeather(x, path)
  invisible(x)
}
