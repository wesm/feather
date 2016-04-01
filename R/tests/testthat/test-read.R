context("read")

test_that("multiplication works", {
  iris_example <- feather_example("iris.feather")
  expect_identical(read_feather(iris_example), as_data_frame(iris))
  expect_identical(read_feather(iris_example, columns = 1:3),
                   as_data_frame(iris)[1:3])
  expect_identical(read_feather(iris_example, columns = "Species"),
                   as_data_frame(iris)["Species"])
})
