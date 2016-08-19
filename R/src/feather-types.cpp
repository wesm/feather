#include <Rcpp.h>
using namespace Rcpp;

#include "feather-types.h"
using namespace feather;

RColType toRColType(FeatherPrimitiveType x) {
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
  }
  throw std::runtime_error("Invalid FeatherColType");
}

RColType toRColType(FeatherColumnType col_type, FeatherPrimitiveType primitive_type) {
  switch(col_type) {
  case feather::ColumnType::PRIMITIVE:
    return toRColType(primitive_type);
  case feather::ColumnType::CATEGORY:
    return R_FACTOR;
  case feather::ColumnType::TIMESTAMP:
    return R_DATETIME;
  case feather::ColumnType::DATE:
    return R_DATE;
  case feather::ColumnType::TIME:
    return R_TIME;
  }
  throw std::runtime_error("Invalid RColType");
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
  throw std::runtime_error("Invalid RColType");
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
  throw std::runtime_error("Invalid RColType");
}


template <class PrimType, class DestType>
void copyRecast(const PrimitiveArray* src, DestType* dest) {
  int n = src->length;

  auto recast = reinterpret_cast<const PrimType*>(src->values);
  std::copy(&recast[0], &recast[0] + n, dest);
}

void setMissing(SEXP x, const PrimitiveArray* val) {
  if (val->null_count == 0)
    return;

  int64_t n = val->length;
  for (int i = 0; i < n; ++i) {
    if (util::bit_not_set(val->nulls, i)) {
      switch(TYPEOF(x)) {
      case LGLSXP: INTEGER(x)[i] = NA_LOGICAL; break;
      case INTSXP: INTEGER(x)[i] = NA_INTEGER; break;
      case REALSXP: REAL(x)[i] = NA_REAL; break;
      case STRSXP: SET_STRING_ELT(x, i, NA_STRING); break;
      default: break;
      }
    }
  }
}

SEXP toSEXP(const PrimitiveArray* val) {
  int64_t n = val->length;
  RColType rType = toRColType(val->type);
  SEXP out = PROTECT(Rf_allocVector(toSEXPTYPE(rType), n));

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

      SEXP raw = PROTECT(Rf_allocVector(RAWSXP, n));
      memcpy(RAW(out), asChar + offset1, n);
      SET_VECTOR_ELT(out, i, raw);
      UNPROTECT(1);
    }
    break;
  }
  default:
    break;
  }

  setMissing(out, val);

  UNPROTECT(1);
  return out;
}

int64_t timeScale(TimeUnit::type unit) {
  switch(unit) {
  case TimeUnit::SECOND:      return 1;
  case TimeUnit::MILLISECOND: return 1e3;
  case TimeUnit::MICROSECOND: return 1e6;
  case TimeUnit::NANOSECOND:  return 1e9;
  }
  throw std::runtime_error("Invalid TimeUnit");
}

// Used to convert INT64 TIME and TIMESTAMP to a double vector.
// This loses precision, but I don't know of a better approach given
// the absense of an INT64 type in R.
SEXP rescaleFromInt64(const PrimitiveArray* pArray, double scale = 1) {
  if (pArray->type != PrimitiveType::INT64)
    stop("Not an INT64");

  auto pValues = reinterpret_cast<const int64_t*>(pArray->values);
  int n = pArray->length;

  SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
  double* pOut = REAL(out);

  if (scale == 1) {
    std::copy(&pValues[0], &pValues[0] + n, pOut);
  } else {
    for (int i = 0; i < n; ++i) {
      pOut[i] = pValues[i] / scale;
    }
  }
  setMissing(out, pArray);

  UNPROTECT(1);
  return out;
}

template <typename T>
static void write_factor_codes(const PrimitiveArray* arr, int* out) {
  auto codes = reinterpret_cast<const T*>(arr->values);
  if (arr->null_count > 0) {
    for (int i = 0; i < arr->length; ++i) {
      // The bit is 1 if it is not null
      out[i] = util::get_bit(arr->nulls, i) ? codes[i] + 1 : NA_INTEGER;
    }
  } else {
    for (int i = 0; i < arr->length; ++i) {
      out[i] = codes[i] + 1;
    }
  }
}

SEXP toSEXP(const ColumnPtr& x) {
  ColumnMetadataPtr meta = x->metadata();
  const PrimitiveArray* val(&x->values());

  switch(x->type()) {
  case feather::ColumnType::PRIMITIVE:
    return toSEXP(val);
  case feather::ColumnType::CATEGORY: {
    IntegerVector out(val->length);

    // Add 1 to category values
    switch (val->type) {
      case PrimitiveType::INT8:
        write_factor_codes<int8_t>(val, INTEGER(out));
        break;
      case PrimitiveType::INT16:
        write_factor_codes<int16_t>(val, INTEGER(out));
        break;
      case PrimitiveType::INT32:
        write_factor_codes<int32_t>(val, INTEGER(out));
        break;
      case PrimitiveType::INT64:
        write_factor_codes<int64_t>(val, INTEGER(out));
        break;
      default:
        stop("Factor codes not a signed integer");
        break;
    }

    auto x_cat = static_cast<feather::CategoryColumn*>(x.get());
    const PrimitiveArray* levels(&x_cat->levels());

    out.attr("levels") = Rf_coerceVector(toSEXP(levels), STRSXP);
    out.attr("class") = "factor";
    return out;
  }
  case feather::ColumnType::DATE: {
    IntegerVector out = toSEXP(val);
    out.attr("class") = "Date";
    return out;
  }
  case feather::ColumnType::TIME: {
    auto x_time = static_cast<feather::TimeColumn*>(x.get());

    DoubleVector out = rescaleFromInt64(val, timeScale(x_time->unit()));
    out.attr("class") = CharacterVector::create("hms", "difftime");
    out.attr("units") = "secs";
    return out;
  }
  case feather::ColumnType::TIMESTAMP: {
    auto x_time = static_cast<feather::TimestampColumn*>(x.get());

    DoubleVector out = rescaleFromInt64(val, timeScale(x_time->unit()));
    out.attr("class") = CharacterVector::create("POSIXct", "POSIXt");
    out.attr("tzone") = x_time->timezone();
    return out;


  } //case
  } //switch
  throw std::runtime_error("Not supported yet");
}
