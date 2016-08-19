#include <Rcpp.h>
#include "feather/api.h"

typedef feather::PrimitiveType::type FeatherPrimitiveType;
typedef feather::ColumnType::type FeatherColumnType;
typedef std::unique_ptr<feather::Column> ColumnPtr;
typedef std::shared_ptr<feather::metadata::Column> ColumnMetadataPtr;

enum RColType {
  R_LGL,
  R_INT,
  R_DBL,
  R_CHR,
  R_RAW,
  R_FACTOR,
  R_DATE,
  R_DATETIME,
  R_TIME
};

RColType toRColType(FeatherPrimitiveType x);
RColType toRColType(FeatherColumnType column_type,
    FeatherPrimitiveType primitive_type);

std::string toString(RColType x);
SEXP toSEXP(const ColumnPtr& x);
