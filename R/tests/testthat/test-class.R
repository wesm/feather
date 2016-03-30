context("class")

mtcars.f <- feather(feather_example("mtcars.feather"))
iris.f <- feather(feather_example("iris.feather"))

mtcars.tbl <- tibble::remove_rownames(tibble::as_data_frame(mtcars))
iris.tbl <- tibble::as_data_frame(iris)

test_that("basic access", {
  expect_equal(nrow(mtcars.f), nrow(mtcars))
  expect_equal(ncol(mtcars.f), ncol(mtcars))
  expect_equal(dim(mtcars.f), dim(mtcars))
  expect_equal(dimnames(iris.f), dimnames(iris))
  expect_equal(colnames(iris.f), colnames(iris))
  expect_equal(iris.f[1:5, 1:5], iris.tbl[1:5, 1:5])
  expect_equal(iris.f[, 1:5], iris.tbl[, 1:5])
  expect_equal(iris.f[1:5, ], iris.tbl[1:5, ])
  expect_equal(iris.f[1:5], iris.tbl[1:5])
  expect_equal(iris.f[], iris.tbl[])
  expect_equal(tibble::as_data_frame(iris.f), iris.tbl)
  expect_equal(as.data.frame(iris.f), iris)
})

test_that("invalid column indexes", {
  expect_error(iris.f[-3], "Only positive")
  expect_error(iris.f[0], "Only positive")
  expect_error(iris.f[-3:3], "Only positive")
  expect_error(iris.f[6], "bounds")
  expect_error(iris.f[3:6], "bounds")
})
