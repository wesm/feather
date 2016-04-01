// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "feather_types.h"
#include <Rcpp.h>

using namespace Rcpp;

// metadataFeather
List metadataFeather(const std::string& path);
RcppExport SEXP feather_metadataFeather(SEXP pathSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const std::string& >::type path(pathSEXP);
    __result = Rcpp::wrap(metadataFeather(path));
    return __result;
END_RCPP
}
// openFeather
XPtr<feather::TableReader> openFeather(const std::string& path);
RcppExport SEXP feather_openFeather(SEXP pathSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const std::string& >::type path(pathSEXP);
    __result = Rcpp::wrap(openFeather(path));
    return __result;
END_RCPP
}
// colsFeather
double colsFeather(List feather);
RcppExport SEXP feather_colsFeather(SEXP featherSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< List >::type feather(featherSEXP);
    __result = Rcpp::wrap(colsFeather(feather));
    return __result;
END_RCPP
}
// rowsFeather
double rowsFeather(List feather);
RcppExport SEXP feather_rowsFeather(SEXP featherSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< List >::type feather(featherSEXP);
    __result = Rcpp::wrap(rowsFeather(feather));
    return __result;
END_RCPP
}
// colnamesFeather
CharacterVector colnamesFeather(List feather);
RcppExport SEXP feather_colnamesFeather(SEXP featherSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< List >::type feather(featherSEXP);
    __result = Rcpp::wrap(colnamesFeather(feather));
    return __result;
END_RCPP
}
// coldataFeather
List coldataFeather(List feather, IntegerVector indexes);
RcppExport SEXP feather_coldataFeather(SEXP featherSEXP, SEXP indexesSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< List >::type feather(featherSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type indexes(indexesSEXP);
    __result = Rcpp::wrap(coldataFeather(feather, indexes));
    return __result;
END_RCPP
}
// writeFeather
void writeFeather(DataFrame df, const std::string& path);
RcppExport SEXP feather_writeFeather(SEXP dfSEXP, SEXP pathSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< DataFrame >::type df(dfSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type path(pathSEXP);
    writeFeather(df, path);
    return R_NilValue;
END_RCPP
}
