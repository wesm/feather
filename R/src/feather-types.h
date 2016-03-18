#include <Rcpp.h>
#include "feather/api.h"

typedef feather::PrimitiveType::type FeatherColType;

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
FeatherColType toFeatherColType(RColType x);
std::string toString(RColType x);
