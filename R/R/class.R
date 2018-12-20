#' Access a feather store like a data frame
#'
#' These functions permit using a feather dataset much like a regular
#' (read-only) data frame.
#'
#' @inheritParams read_feather
#' @return An object of class \code{feather}
#' @export
feather <- function(path) {
  path <- enc2native(normalizePath(path, mustWork = TRUE))

  openFeather(path)
}

#' @export
close.feather <- function(con, ...) {
  closeFeather(con)
}

#' @export
row.names.feather <- function(x) as.character(seq_len(nrow(x)))

#' @export
dimnames.feather <- function(x) list(row.names(x), names(x))

#' @export
dim.feather <- function(x) c(rowsFeather(x), length(x))

.column_indexes_feather <- function(x, j) {
  if( is.character(j) ) {
    wrong <- !(j %in% colnames(x))
    if (any(wrong)) {
      names <- j[wrong]
      stop(sprintf("undefined columns: %s", paste(names, collapse = ", ")))
    }
    j <- match(j, colnames(x))
  } else if (is.logical(j)) {
    j <- which(j)
  } else if (is.numeric(j)) {
    if (any(j <= 0)) {
      stop("Only positive column indexes supported.", call. = FALSE)
    } else if (any(j > ncol(x))) {
      stop("Subscript out of bounds.", call. = FALSE)
    }
  } else {
    stop("Can't use ", class(x), " for column indexes.", call. = FALSE)
  }
  j
}

#' @export
`[[.feather` <- function(x, i, exact = TRUE) {
  if (is.character(i) && length(i) == 1L && !(i %in% names(x))) {
    stop("Unknown column '", i, "'", call. = FALSE)
  }
  if (!exact) {
    warning("exact ignored", call. = FALSE)
  }

  x[i][[1L]]
}

#' @export
`$.feather` <- function(x, i) {
  if (is.character(i) && !(i %in% names(x))) {
    stop("Unknown column '", i, "'", call. = FALSE)
  }


  x[[i]]
}

#' @export
`[.feather` <- function(x, i, j, drop = FALSE) {
  if (drop) warning("drop ignored", call. = FALSE)

  nr <- nrow(x)

  # Escape early if nargs() == 2L; ie, column subsetting
  if (nargs() <= 2) {
    if (missing(i))
      i <- seq_len(ncol(x))
    else
      i <- .column_indexes_feather(x, i)
    return(coldataFeather(x, i))
  }

  # First, subset columns
  if (!missing(j)) {
    j <- .column_indexes_feather(x, j)
    df <- coldataFeather(x, j)
  } else {
    df <- coldataFeather(x, seq_len(ncol(x)))
  }

  if (missing(i))
    df
  else
    df[i, ]
}


# Coercion ----------------------------------------------------------------

#' @export
#' @importFrom tibble as_tibble
as.data.frame.feather <- function(x, row.names = NULL, optional = FALSE, ...) {
  if (!is.null(row.names))
    stop("row.names must be NULL for as.data.frame.feather")
  as.data.frame(as_tibble(x[]))
}


#' @export
as.list.feather <- function(x, ...) {
  as.list(as_tibble(x))
}


# Output ------------------------------------------------------------------

#' @export
print.feather <- function(x, ...) {
  cat("Source: feather store ", dim_desc(dim(x)), "\n\n", sep = "")
  print(tibble::trunc_mat(as.data.frame(x)))
  invisible(x)
}
