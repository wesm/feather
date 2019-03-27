context("overwrite")

test_that("can read new data", {
  path <- tempfile()

  x <- 1:100
  write_feather(tibble::tibble(x = x), path)
  on.exit(file.remove(path))
  res <- read_feather(path)

  y <- x[1:(length(x)/2)]
  write_feather(tibble::tibble(y = y), path)
  resy <- read_feather(path)
  expect_identical(resy$y, y)
})
