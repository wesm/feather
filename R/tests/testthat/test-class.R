context("class")

mtcars.f <- feather(feather_example("mtcars.feather"))
iris.f <- feather(feather_example("iris.feather"))

mtcars.tbl <- tibble::remove_rownames(tibble::as_tibble(mtcars))
iris.tbl <- tibble::as_tibble(iris)

test_that("basic access", {
  expect_equal(nrow(mtcars.f), nrow(mtcars))
  expect_equal(ncol(mtcars.f), ncol(mtcars))
  expect_equal(dim(mtcars.f), dim(mtcars))
  expect_equal(dimnames(iris.f), dimnames(iris))
  expect_equal(colnames(iris.f), colnames(iris))
  expect_equal(names(iris.f), names(iris))
  expect_identical(iris.f[1:5, 1:5], iris.tbl[1:5, 1:5])
  expect_identical(iris.f[, 1:5], iris.tbl[, 1:5])
  expect_identical(iris.f[1:5, ], iris.tbl[1:5, ])
  expect_identical(iris.f[1:5], iris.tbl[1:5])
  expect_identical(iris.f[], iris.tbl[])
  expect_identical(iris.f[["Species"]], iris.tbl[["Species"]])
  expect_identical(iris.f[["Sepal.Length"]], iris.tbl[["Sepal.Length"]])
  expect_identical(iris.f$Species, iris.tbl$Species)
  expect_identical(iris.f$Sepal.Width, iris.tbl$Sepal.Width)
})

test_that("coercion", {
  expect_identical(tibble::as_tibble(iris.f), iris.tbl)
  expect_identical(as.data.frame(iris.f), iris)
  expect_identical(as.list(iris.f), as.list(iris))
})

test_that("invalid column indexes", {
  expect_error(iris.f[-3], "Only positive")
  expect_error(iris.f[0], "Only positive")
  expect_error(iris.f[-3:3], "Only positive")
  expect_error(iris.f[6], "bounds")
  expect_error(iris.f[3:6], "bounds")
})

test_that("close", {
  iris.f2 <- feather(feather_example("iris.feather"))
  expect_equal(nrow(iris.f2), 150L)
  close(iris.f2)
  expect_error(nrow(iris.f2), "feather already closed")
})
