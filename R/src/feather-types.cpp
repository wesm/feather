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


template <class PrimType, class DestType>
void copyRecast(const PrimitiveArray* src, DestType* dest) {
  int n = src->length;

  auto recast = reinterpret_cast<const PrimType*>(src->values);
  std::copy(&recast[0], &recast[0] + n, dest);
}

SEXP toSEXP(const PrimitiveArray* val) {
  int64_t n = val->length;
  RColType rType = toRColType(val->type);
  SEXP out = Rf_allocVector(toSEXPTYPE(rType), n);

  switch(val->type) {
  case PrimitiveType::BOOL: {
    for (int i = 0; i < n; ++i) {
      INTEGER(out)[i] = util::get_bit(val->values, i);
    }
    break;
  }
  case PrimitiveType::INT8:
    copyRecast<int8_t>(val, INTEGER(out));
    break;
  case PrimitiveType::INT16:
    copyRecast<int16_t>(val, INTEGER(out));
    break;
  case PrimitiveType::INT32:
    copyRecast<int32_t>(val, INTEGER(out));
    break;
  case PrimitiveType::UINT8:
    copyRecast<uint8_t>(val, INTEGER(out));
    break;
  case PrimitiveType::UINT16:
    copyRecast<uint16_t>(val, INTEGER(out));
    break;
  case PrimitiveType::UINT32:
    copyRecast<uint32_t>(val, INTEGER(out));
    break;
  case PrimitiveType::INT64:
    Rf_warningcall(R_NilValue, "Coercing int64 to double");
    copyRecast<int64_t>(val, REAL(out));
    break;
  case PrimitiveType::UINT64:
    Rf_warningcall(R_NilValue, "Coercing uint64 to double");
    copyRecast<int64_t>(val, REAL(out));
    break;
  case PrimitiveType::FLOAT:
    copyRecast<float>(val, REAL(out));
    break;
  case PrimitiveType::DOUBLE:
    copyRecast<double>(val, REAL(out));
    break;

  case PrimitiveType::UTF8: {
    auto asChar = reinterpret_cast<const char*>(val->values);
    for (int i = 0; i < n; ++i) {
      uint32_t offset1 = val->offsets[i], offset2 = val->offsets[i + 1];
      SEXP string = Rf_mkCharLenCE(asChar + offset1, offset2 - offset1, CE_UTF8);
      SET_STRING_ELT(out, i, string);
    }
    break;
  }
  case PrimitiveType::BINARY: {
    auto asChar = reinterpret_cast<const char*>(val->values);
    for (int i = 0; i < n; ++i) {
      uint32_t offset1 = val->offsets[i], offset2 = val->offsets[i + 1];
      int32_t n = offset2 - offset1;

      SEXP raw = Rf_allocVector(RAWSXP, n);
      memcpy(RAW(out), asChar + offset1, n);
      SET_VECTOR_ELT(out, i, raw);
    }
    break;
  }
  default:
    break;
  }

  if (val->null_count > 0) {
    for (int i = 0; i < n; ++i) {
      if (util::get_bit(val->nulls, i)) {
        switch(TYPEOF(out)) {
        case LGLSXP: INTEGER(out)[i] = NA_LOGICAL; break;
        case INTSXP: INTEGER(out)[i] = NA_INTEGER; break;
        case REALSXP: REAL(out)[i] = NA_REAL; break;
        case STRSXP: SET_STRING_ELT(out, i, NA_STRING); break;
        default: break;
        }
      }
    }
  }


  return out;
}

SEXP toSEXP(ColumnPtr x) {
  ColumnMetadataPtr meta = x->metadata();
  const PrimitiveArray* val(&x->values());

  switch(x->type()) {
  case feather::ColumnType::PRIMITIVE:
    return toSEXP(val);
  case feather::ColumnType::CATEGORY: {
    IntegerVector out = toSEXP(val);

    auto x_cat = std::static_pointer_cast<feather::CategoryColumn>(x);
    const PrimitiveArray* levels(&x_cat->levels());

    out.attr("levels") = toSEXP(levels);
    out.attr("class") = "factor";
    return out;
  }
  case feather::ColumnType::TIMESTAMP:
  case feather::ColumnType::DATE:
  case feather::ColumnType::TIME:
    stop("Not supported yet");
    return 0;
  }
}
