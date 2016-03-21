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
  case PrimitiveType::UINT8:
  case PrimitiveType::UINT16:
  case PrimitiveType::UINT32:
    return R_INT;
  case PrimitiveType::INT64:
  case PrimitiveType::UINT64:
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

RColType toRColType(ColumnPtr x) {
  switch(x->type()) {
  case feather::ColumnType::PRIMITIVE:
    return toRColType(x->metadata()->values().type);
  case feather::ColumnType::CATEGORY:
    return R_FACTOR;
  case feather::ColumnType::TIMESTAMP:
    return R_DATETIME;
  case feather::ColumnType::DATE:
    return R_DATE;
  case feather::ColumnType::TIME:
    return R_TIME;
  }
}

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

SEXPTYPE toSEXPTYPE(RColType x) {
  switch(x) {
  case R_LGL:      return LGLSXP;
  case R_INT:      return INTSXP;
  case R_DBL:      return REALSXP;
  case R_CHR:      return STRSXP;
  case R_RAW:      return VECSXP;
  case R_FACTOR:   return INTSXP;
  case R_DATE:     return INTSXP;
  case R_DATETIME: return REALSXP;
  case R_TIME:     return REALSXP;
  }
}


SEXP toSEXP(ColumnPtr x) {
  ColumnMetadataPtr meta = x->metadata();
  const PrimitiveArray* val(&x->values());
  int64_t n = val->length;

  RColType rType = toRColType(meta->values().type);
  SEXP out = Rf_allocVector(toSEXPTYPE(rType), n);

  switch(meta->values().type) {
  case PrimitiveType::BOOL: {
    for (int i = 0; i < n; ++i) {
      INTEGER(out)[i] = util::get_bit(val->values, i);
    }
    break;
  }
  case PrimitiveType::INT8: {
    auto int8val = reinterpret_cast<const int8_t*>(val->values);
    for (int i = 0; i < n; ++i) {
      INTEGER(out)[i] = int8val[i];
    }
    break;
  }
  case PrimitiveType::INT16: {
    auto int16val = reinterpret_cast<const int16_t*>(val->values);
    for (int i = 0; i < n; ++i) {
      INTEGER(out)[i] = int16val[i];
    }
    break;
  }
  case PrimitiveType::INT32:
    memcpy(INTEGER(out), val->values, n * ByteSize(val->type));
    break;
  case PrimitiveType::INT64: {
    Rf_warningcall(R_NilValue, "Coercing int64 to double");
    auto int64val = reinterpret_cast<const int64_t*>(val->values);
    for (int i = 0; i < n; ++i) {
      REAL(out)[i] = int64val[i];
    }
    break;
  }
  default:
    break;
  }

  return out;
}
