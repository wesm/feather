roundtrip_vector <- function(x) {
  df <- dplyr::data_frame(x = x)
  roundtrip(df)$x
}

roundtrip <- function(df) {
  temp <- tempfile()
  write_feather(df, temp)
  on.exit(unlink(temp))

  read_feather(temp)
}

