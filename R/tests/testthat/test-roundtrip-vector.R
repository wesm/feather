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
  expect_identical(roundtrip_vector(x), x)
})


test_that("preserves NA in factor and levels", {
  x1 <- factor(c("abc", "def", NA))
  x2 <- addNA(x1)

  expect_identical(roundtrip_vector(x1), x1)
  expect_identical(roundtrip_vector(x2), x2)
})


# Date --------------------------------------------------------------------

test_that("preserves dates", {
  x <- as.Date("2010-01-01") + c(0L, 365L, NA)
  mode(x) <- "integer"
  expect_identical(roundtrip_vector(x), x)
})

# Time --------------------------------------------------------------------

test_that("preserves hms", {
  x <- hms::hms(1:100)
  expect_identical(roundtrip_vector(x), x)
})

test_that("converts time to hms", {
  x1 <- structure(1:100, class = "time")
  x2 <- roundtrip_vector(x1)

  expect_s3_class(x2, "hms")
})


# Timestamp/POSIXct -------------------------------------------------------

test_that("preserves times", {
  x1 <- ISOdate(2001, 10, 10, tz = "US/Pacific") + c(0, NA)
  x2 <- roundtrip_vector(x1)

  expect_identical(attr(x1, "tzone"), attr(x2, "tzone"))
  expect_identical(attr(x1, "class"), attr(x1, "class"))
  expect_identical(unclass(x1), unclass(x2))
})

test_that("throws error on POSIXlt", {
  df <- data.frame(x = Sys.time())
  df$x <- as.POSIXlt(df$x)

  expect_error(roundtrip(df), "Can not write POSIXlt")
})


test_that("doesn't lose undue precision", {
  base <- ISOdate(2001, 10, 10)
  x1 <- base + 1e-6 * (0:3)
  x2 <- roundtrip_vector(x1)

  expect_identical(x1, x2)
})
