#include <Rcpp.h>
using namespace Rcpp;

#include "feather/api.h"
using namespace feather;

// [[Rcpp::export]]
IntegerVector feather_dim(std::string path) {
  std::string fullPath(R_ExpandFileName(path.c_str()));
  std::unique_ptr<TableReader> table;

  if (!TableReader::OpenFile(fullPath, &table).ok()) {
    stop("Failed to open '%s'", path);
  }
  return IntegerVector::create(table->num_rows(), table->num_columns());
}

// [[Rcpp::export]]
CharacterVector feather_names(std::string path) {
  std::string fullPath(R_ExpandFileName(path.c_str()));
  std::unique_ptr<TableReader> table;

  if (!TableReader::OpenFile(fullPath, &table).ok()) {
    stop("Failed to open '%s'", path);
  }

  int n = table->num_columns();
  CharacterVector out(n);

  for (int i = 0; i < n; ++i) {
    std::shared_ptr<Column> col;
    if (!table->GetColumn(i, &col).ok()) {
      stop("Failed to retrieve column %i", i);
    }

    out[i] = col->name();
  }

  return out;
}
