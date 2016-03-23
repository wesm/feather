#include <Rcpp.h>
#include "feather/api.h"

inline void stopOnFailure(feather::Status st) {
  if (st.ok())
    return;

  Rcpp::stop(st.ToString());
}
