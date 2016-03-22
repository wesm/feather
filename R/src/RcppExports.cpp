// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// dimFeather
IntegerVector dimFeather(std::string path);
RcppExport SEXP feather_dimFeather(SEXP pathSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< std::string >::type path(pathSEXP);
    __result = Rcpp::wrap(dimFeather(path));
    return __result;
END_RCPP
}
// metadataFeather
CharacterVector metadataFeather(std::string path);
RcppExport SEXP feather_metadataFeather(SEXP pathSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< std::string >::type path(pathSEXP);
    __result = Rcpp::wrap(metadataFeather(path));
    return __result;
END_RCPP
}
// readFeather
List readFeather(std::string path);
RcppExport SEXP feather_readFeather(SEXP pathSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< std::string >::type path(pathSEXP);
    __result = Rcpp::wrap(readFeather(path));
    return __result;
END_RCPP
}
// writeFeather
void writeFeather(DataFrame df, std::string path);
RcppExport SEXP feather_writeFeather(SEXP dfSEXP, SEXP pathSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< DataFrame >::type df(dfSEXP);
    Rcpp::traits::input_parameter< std::string >::type path(pathSEXP);
    writeFeather(df, path);
    return R_NilValue;
END_RCPP
}
