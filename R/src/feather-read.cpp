#include <Rcpp.h>
using namespace Rcpp;

#include "feather/api.h"
using namespace feather;

#include "feather-types.h"

std::unique_ptr<TableReader> openFeatherTable(const std::string& path) {
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
List metadataFeather(const std::string& path) {
  auto table = openFeatherTable(path);

  int n = table->num_rows(), p = table->num_columns();
  CharacterVector types(p), names(p);

  for (int j = 0; j < p; ++j) {
    auto col = getColumn(table, j);

    names[j] = col->name();
    types[j] = toString(toRColType(col));
  }
  types.attr("names") = names;

  auto out = List::create(
    _["path"] = path,
    _["dim"] = IntegerVector::create(n, p),
    _["types"] = types,
    _["description"] = table->GetDescription()
  );
  out.attr("class") = "feather_metadata";
  return out;
}

// [[Rcpp::export]]
List readFeather(const std::string& path) {
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


