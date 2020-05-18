context("class")

mtcars.f <- feather(feather_example("mtcars.feather"))
iris.f <- feather(feather_example("iris.feather"))

mtcars.tbl <- tibble::remove_rownames(tibble::as_tibble(mtcars))
iris.tbl <- tibble::as_tibble(iris)

test_that("basic access", {
  expect_equal(nrow(mtcars.f), nrow(mtcars))
  expect_equal(ncol(mtcars.f), ncol(mtcars))
  expect_equal(dim(mtcars.f), dim(mtcars))
  # TODO(arrow): implement these methods
  # expect_equal(dimnames(iris.f), dimnames(iris))
  # expect_equal(colnames(iris.f), colnames(iris))
  expect_equal(names(iris.f), names(iris))
  expect_identical(as.data.frame(iris.f[1:5, 1:5]), iris.tbl[1:5, 1:5])
  expect_identical(as.data.frame(iris.f[, 1:5]), iris.tbl[, 1:5])
  expect_identical(as.data.frame(iris.f[1:5, ]), iris.tbl[1:5, ])
  # TODO(arrow): implement nargs <= 2
  # expect_identical(as.data.frame(iris.f[1:5]), iris.tbl[1:5])
  expect_identical(as.data.frame(iris.f[]), iris.tbl[])
  expect_identical(as.vector(iris.f[["Species"]]), iris.tbl[["Species"]])
  expect_identical(as.vector(iris.f[["Sepal.Length"]]), iris.tbl[["Sepal.Length"]])
  expect_identical(as.vector(iris.f$Species), iris.tbl$Species)
  expect_identical(as.vector(iris.f$Sepal.Width), iris.tbl$Sepal.Width)
})

test_that("coercion", {
  expect_identical(tibble::as_tibble(iris.f), iris.tbl)
  expect_identical(as.data.frame(iris.f), iris.tbl)
  skip("TODO(arrow): implement as.list.Table")
  expect_identical(as.list(iris.f), as.list(iris))
})

test_that("invalid column indexes", {
  skip("TODO(arrow): validate these in arrow")
  expect_error(iris.f[-3], "Only positive")
  expect_error(iris.f[0], "Only positive")
  expect_error(iris.f[-3:3], "Only positive")
  expect_error(iris.f[6], "bounds")
  expect_error(iris.f[3:6], "bounds")
})
