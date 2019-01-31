#' @useDynLib feather, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom tibble tibble
NULL

#' Read and write feather files.
#'
#' @param path Path to feather file
#' @param columns Columns to read (names or indexes). Default: Read all columns.
#' @return Both functions return a tibble/data frame. \code{write_feather}
#'   invisibly returns \code{x} (so you can use this function in a pipeline).
#' @examples
#' mtcars2 <- read_feather(feather_example("mtcars.feather"))
#' mtcars2
#' @export
read_feather <- function(path, columns = NULL) {
  data <- feather(path)
  on.exit(close(data), add = TRUE)

  if (is.null(columns))
    as_tibble(data)
  else
    as_tibble(data[columns])
}

#' @rdname read_feather
#' @param x A data frame to write to disk
#' @export
write_feather <- function(x, path) {
  if (!is.data.frame(x)) {
    stop("`x` must be a data frame", call. = FALSE)
  }
  writeFeather(x, path)
  invisible(x)
}

#' Retrieve metadata about a feather file
#'
#' Returns the dimensions, field names, and types; and optional dataset
#' description.
#'
#' @param path Path to feather file
#' @return A list with class "feather_metadata".
#' @export
feather_metadata <- function(path) {
  path <- path.expand(path)
  metadataFeather(path)
}

#' @export
print.feather_metadata <- function(x, ...) {
  cat("<Feather file>\n")
  cat(dim_desc(x$dim), " @ ", x$path, "\n", sep = "")

  names <- format(encodeString(names(x$types), quote = "'"))
  cat(paste0("* ", names, ": ", x$types, "\n"), sep = "")
}


# Force import of hms package
#' @importFrom hms hms
NULL
