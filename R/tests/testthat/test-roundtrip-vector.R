context("roundtrip-vector")

# Logical -----------------------------------------------------------------

test_that("preserves three logical values", {
  x <- c(FALSE, TRUE, NA)
  expect_identical(roundtrip_vector(x), x)
})

test_that("preserves logical vector of length 0", {
  x <- logical()
  expect_identical(roundtrip_vector(x), x)
})

# Integer ----------------------------------------------------------------

test_that("preserves integer values", {
  x <- 1:10
  expect_identical(roundtrip_vector(x), x)
})

# Double -----------------------------------------------------------------

test_that("preserves special floating point values", {
  x <- c(Inf, -Inf, NaN, NA)
  expect_identical(roundtrip_vector(x), x)
})

test_that("doesn't lose precision", {
  x <- c(1/3, sqrt(2), pi)
  expect_identical(roundtrip_vector(x), x)
})

# Character ---------------------------------------------------------------

test_that("preserves character values", {
  x <- c("this is a string", "", NA, "another string")
  expect_identical(roundtrip_vector(x), x)
})

test_that("can have NA on end of string", {
  x <- c("this is a string", NA)
  expect_identical(roundtrip_vector(x), x)
})

test_that("always coerces to UTF-8", {
  x <- iconv("é", to = "latin1")
  y <- roundtrip_vector(x)

  expect_identical(x, y) # string comparison always re-encodes first
  expect_identical(Encoding(x), "latin1")
  expect_identical(Encoding(y), "UTF-8")
})


# Factor ------------------------------------------------------------------

test_that("preserves simple factor", {
  x <- factor(c("abc", "def"))
  expect_equal(roundtrip_vector(x), x)
})


test_that("preserves NA in factor and levels", {
  x1 <- factor(c("abc", "def", NA))
  x2 <- addNA(x1)

  expect_equal(roundtrip_vector(x1), x1)
  expect_equal(roundtrip_vector(x2), x2)
})

