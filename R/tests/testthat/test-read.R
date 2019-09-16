context("read")

test_that("iris example is identical to iris data frame", {
  iris_example <- feather_example("iris.feather")
  expect_identical(read_feather(iris_example), as_tibble(iris))
  expect_identical(read_feather(iris_example, columns = 1:3),
                   as_tibble(iris)[1:3])
  expect_identical(read_feather(iris_example, columns = "Species"),
                   as_tibble(iris)["Species"])
})


test_that("can read/write with utf-8 filename", {
  # Fails on windows and in strict latin1 locale
  skip_on_cran()

  path <- file.path(tempdir(), "\u00e5.feather") # å
  on.exit(file.remove(path))

  write_feather(mtcars, path)

  mtcars2 <- read_feather(path)
  expect_equal(dim(mtcars2), c(32, 11))
})
