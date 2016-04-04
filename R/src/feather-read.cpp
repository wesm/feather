#include <Rcpp.h>
using namespace Rcpp;

#include "feather/api.h"
using namespace feather;

#include "feather-types.h"
#include "feather-utils.h"

std::unique_ptr<TableReader> openFeatherTable(const std::string& path) {
  std::unique_ptr<TableReader> table;
  std::string fullPath(R_ExpandFileName(path.c_str()));

  stopOnFailure(TableReader::OpenFile(fullPath, &table));
  return table;
}

std::unique_ptr<Column> getColumn(const TableReader& table, int i) {
  std::unique_ptr<Column> col;
  stopOnFailure(table.GetColumn(i, &col));
  return col;
}

// [[Rcpp::export]]
List metadataFeather(const std::string& path) {
  auto table = openFeatherTable(path);

  int n = table->num_rows(), p = table->num_columns();
  CharacterVector types(p), names(p);

  for (int j = 0; j < p; ++j) {
    auto col = getColumn(*table, j);

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


CharacterVector colnamesAsCharacterVector(const TableReader& table) {
  int n = table.num_columns();
  CharacterVector names(n);

  for (int i = 0; i < n; ++i) {
    auto col = getColumn(table, i);

    names[i] = col->name();
  }

  return names;
}


// [[Rcpp::export]]
List openFeather(const std::string& path) {
  auto table = openFeatherTable(path);

  int n = table->num_columns();
  List out(n);

  out.attr("names") = colnamesAsCharacterVector(*table);
  out.attr("table") = XPtr<TableReader>(table.release(), true);
  out.attr("class") = "feather";
  return out;
}


TableReader* getTableFromFeather(const List& feather) {
  return Rcpp::as<XPtr<TableReader> >(feather.attr("table"));
}


// [[Rcpp::export]]
double rowsFeather(const List& feather) {
  auto table = getTableFromFeather(feather);
  return (double)table->num_rows();
}


// [[Rcpp::export]]
List coldataFeather(const List& feather, const IntegerVector& indexes) {
  auto table = getTableFromFeather(feather);

  int n = indexes.length(), p = table->num_rows();
  List out(n), names(n);

  for (int i = 0; i < n; ++i) {
    auto col = getColumn(*table, indexes[i] - 1);

    names[i] = col->name();
    out[i] = toSEXP(col);
  }

  out.attr("names") = names;
  out.attr("row.names") = IntegerVector::create(NA_INTEGER, -p);
  out.attr("class") = CharacterVector::create("tbl_df", "tbl", "data.frame");
  return out;
}
