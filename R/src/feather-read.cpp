#include <Rcpp.h>
using namespace Rcpp;

#include "feather/api.h"
using namespace feather;

#include "feather-types.h"
#include "feather-utils.h"

std::unique_ptr<TableReader> openFeatherTable(const std::string& path) {
  std::unique_ptr<TableReader> table;

  stopOnFailure(TableReader::OpenFile(path.c_str(), &table));
  return table;
}

std::unique_ptr<Column> getColumn(const TableReader& table, int i) {
  std::unique_ptr<Column> col;
  stopOnFailure(table.GetColumn(i, &col));
  return col;
}

std::shared_ptr<metadata::Column> getColumnMetadata(const TableReader& table,
    int i) {
  std::shared_ptr<metadata::Column> meta;
  stopOnFailure(table.GetColumnMetadata(i, &meta));
  return meta;
}

// [[Rcpp::export]]
List metadataFeather(const std::string& path) {
  std::unique_ptr<TableReader> table = openFeatherTable(path);

  int n = table->num_rows(), p = table->num_columns();
  CharacterVector types(p), names(p);

  for (int j = 0; j < p; ++j) {
    auto meta = getColumnMetadata(*table, j);

    names[j] = Rf_mkCharCE(meta->name().c_str(), CE_UTF8);
    types[j] = toString(toRColType(meta->type(), meta->values_type()));
  }
  types.attr("names") = names;

  auto out = List::create(
    _["path"] = path,
    _["dim"] = IntegerVector::create(n, p),
    _["types"] = types,
    _["description"] = table->GetDescription(),
    _["version"] = table->version()
  );
  out.attr("class") = "feather_metadata";
  return out;
}


CharacterVector colnamesAsCharacterVector(const TableReader& table) {
  int n = table.num_columns();
  CharacterVector names(n);

  for (int i = 0; i < n; ++i) {
    auto meta = getColumnMetadata(table, i);
    names[i] = Rf_mkCharCE(meta->name().c_str(), CE_UTF8);
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


// [[Rcpp::export]]
void closeFeather(const List& feather) {
  Rcpp::as<XPtr<TableReader> >(feather.attr("table")).release();
}


TableReader* getTableFromFeather(const List& feather) {
  TableReader* table = Rcpp::as<XPtr<TableReader> >(feather.attr("table")).get();
  if (!table) Rcpp::stop("feather already closed");
  return table;
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

    names[i] = Rf_mkCharCE(col->name().c_str(), CE_UTF8);
    out[i] = toSEXP(col);
  }

  out.attr("names") = names;
  out.attr("row.names") = IntegerVector::create(NA_INTEGER, -p);
  out.attr("class") = CharacterVector::create("tbl_df", "tbl", "data.frame");
  return out;
}
