#include <Rcpp.h>
#include "feather/api.h"

typedef feather::PrimitiveType::type FeatherColType;
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

RColType toRColType(FeatherColType x);
RColType toRColType(const ColumnPtr& x);

std::string toString(RColType x);
SEXP toSEXP(const ColumnPtr& x);
