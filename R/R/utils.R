dim_desc <- function(x) {
  paste0("[", paste0(big_mark(x), collapse = " x "), "]")
}

big_mark <- function(x, ...) {
  mark <- if (identical(getOption("OutDec"), ",")) "." else ","
  formatC(x, big.mark = mark, ...)
}
