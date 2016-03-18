#include <Rcpp.h>
using namespace Rcpp;

#include "feather-types.h"
using namespace feather;

RColType toRColType(FeatherColType x) {
  switch(x) {
  case PrimitiveType::BOOL:
    return R_LGL;
  case PrimitiveType::INT8:
  case PrimitiveType::INT16:
  case PrimitiveType::INT32:
  case PrimitiveType::INT64:
  case PrimitiveType::UINT8:
  case PrimitiveType::UINT16:
  case PrimitiveType::UINT32:
  case PrimitiveType::UINT64:
    return R_INT;
  case PrimitiveType::FLOAT:
  case PrimitiveType::DOUBLE:
    return R_DBL;
  case PrimitiveType::UTF8:
    return R_CHR;
  case PrimitiveType::BINARY:
    return R_RAW;
  case PrimitiveType::CATEGORY:
    return R_FACTOR;
  case PrimitiveType::TIMESTAMP:
    return R_DATETIME;
  case PrimitiveType::DATE:
    return R_DATE;
  case PrimitiveType::TIME:
    return R_TIME;
  }
};
FeatherColType toFeatherColType(RColType x) {
  switch(x) {
  case R_LGL:      return PrimitiveType::BOOL;
  case R_INT:      return PrimitiveType::INT32;
  case R_DBL:      return PrimitiveType::DOUBLE;
  case R_CHR:      return PrimitiveType::UTF8;
  case R_RAW:      return PrimitiveType::BINARY;
  case R_FACTOR:   return PrimitiveType::CATEGORY;
  case R_DATE:     return PrimitiveType::DATE;
  case R_DATETIME: return PrimitiveType::TIMESTAMP;
  case R_TIME:     return PrimitiveType::TIME;
  }
}
std::string toString(RColType x) {
  switch(x) {
  case R_LGL:      return "logical";
  case R_INT:      return "integer";
  case R_DBL:      return "double";
  case R_CHR:      return "character";
  case R_RAW:      return "raw-list";
  case R_FACTOR:   return "factor";
  case R_DATE:     return "date";
  case R_DATETIME: return "datetime";
  case R_TIME:     return "time";
  }
}
