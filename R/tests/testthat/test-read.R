context("read")

test_that("multiplication works", {
  iris_example <- feather_example("iris.feather")
  expect_identical(read_feather(iris_example), as_data_frame(iris))
  expect_identical(read_feather(iris_example, columns = 1:3),
                   as_data_frame(iris)[1:3])
  expect_identical(read_feather(iris_example, columns = "Species"),
                   as_data_frame(iris)["Species"])
})


test_that("can read/write with utf-8 filename", {
  path <- file.path(tempdir(), "\u00e5.feather") # å
  on.exit(file.remove(path))

  write_feather(mtcars, path)

  mtcars2 <- read_feather(path)
  expect_equal(dim(mtcars2), c(32, 11))
})
