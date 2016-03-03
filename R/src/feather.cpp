#include <Rcpp.h>
using namespace Rcpp;

#include <feather/api.h>

// [[Rcpp::export]]
std::string magicBytes() {
  return feather::FEATHER_MAGIC_BYTES;
}
