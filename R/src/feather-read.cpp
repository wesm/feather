#include <Rcpp.h>
using namespace Rcpp;

#include "feather/api.h"
using namespace feather;

#include "feather-types.h"

std::unique_ptr<TableReader> openFeatherTable(std::string path) {
  std::unique_ptr<TableReader> table;
  std::string fullPath(R_ExpandFileName(path.c_str()));

  auto st = TableReader::OpenFile(fullPath, &table);
  if (st.ok())
    return table;

  stop("Failed to open '%s' (%s)", path, st.CodeAsString());
  return table; // silence warning
}

// [[Rcpp::export]]
IntegerVector feather_dim(std::string path) {
  auto table = openFeatherTable(path);

  return IntegerVector::create(table->num_rows(), table->num_columns());
}

// [[Rcpp::export]]
CharacterVector feather_metadata(std::string path) {
  auto table = openFeatherTable(path);

  int n = table->num_columns();
  CharacterVector out(n), names(n);
  out.attr("names") = names;

  for (int i = 0; i < n; ++i) {
    std::shared_ptr<Column> col;
    if (!table->GetColumn(i, &col).ok()) {
      stop("Failed to retrieve column %i", i);
    }

    names[i] = col->name();
    out[i] = toString(toRColType(col));
  }

  return out;
}

