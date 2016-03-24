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

std::shared_ptr<Column> getColumn(std::unique_ptr<TableReader>& table, int i) {
  std::shared_ptr<Column> col;

  auto st = table->GetColumn(i, &col);
  if (st.ok())
    return col;

  stop("Failed to retrieve column %i (%s)", i, st.CodeAsString());
  return col; // silence warning
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

  for (int i = 0; i < n; ++i) {
    auto col = getColumn(table, i);

    names[i] = col->name();
    out[i] = toString(toRColType(col));
  }

  out.attr("names") = names;
  return out;
}

// [[Rcpp::export]]
List feather_read(std::string path) {
  auto table = openFeatherTable(path);

  int n = table->num_columns(), p = table->num_rows();
  List out(n), names(n);

  for (int i = 0; i < n; ++i) {
    auto col = getColumn(table, i);

    names[i] = col->name();
    out[i] = toSEXP(col);
  }

  out.attr("names") = names;
  out.attr("row.names") = IntegerVector::create(NA_INTEGER, -p);
  out.attr("class") = CharacterVector::create("tbl_df", "tbl", "data.frame");
  return out;
}


