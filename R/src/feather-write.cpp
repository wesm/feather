#include <Rcpp.h>
using namespace Rcpp;

#include "feather/api.h"
using namespace feather;

#include "feather-types.h"


std::shared_ptr<OwnedMutableBuffer> makeBoolBuffer(int n) {
  int nbytes = util::bytes_for_bits(n);

  auto buffer = std::make_shared<OwnedMutableBuffer>();
  buffer->Resize(nbytes);
  memset(buffer->mutable_data(), 0, nbytes);

  return buffer;
}


// Returns an object with pointers into x, so must not be
// used after x goes away
PrimitiveArray intToPrimitiveArray(SEXP x) {
  int n = Rf_length(x);


  auto null_buffer = makeBoolBuffer(n);
  auto nulls = null_buffer->mutable_data();
  uint32_t n_missing = 0;
  int* px = INTEGER(x);
  for (int i = 0; i < n; ++i) {
    if (px[i] == NA_INTEGER) {
      ++n_missing;
      util::set_bit(nulls, i);
    }
  }


  PrimitiveArray out;
  out.type = PrimitiveType::INT32;
  out.length = n;
  out.values = reinterpret_cast<uint8_t*>(INTEGER(x));

  if (n_missing > 0) {
    out.null_count = n_missing;
    out.buffers.push_back(null_buffer);
    out.nulls = nulls;
  } else {
    out.null_count = 0;
  }

  return out;
}

PrimitiveArray dblToPrimitiveArray(SEXP x) {
  int n = Rf_length(x);

  PrimitiveArray out;
  out.type = PrimitiveType::DOUBLE;
  out.length = n;
  out.null_count = 0;
  out.values = reinterpret_cast<uint8_t*>(REAL(x));

  return out;
}


PrimitiveArray toPrimitiveArray(SEXP x) {
  switch(TYPEOF(x)) {
  case INTSXP: return intToPrimitiveArray(x);
  case REALSXP: return dblToPrimitiveArray(x);
  default:
    stop("Unsupported type (%s)", Rf_type2char(TYPEOF(x)));
    throw 0;
  }
}

// [[Rcpp::export]]
void feather_write(DataFrame df, std::string path) {
  std::unique_ptr<TableWriter> table;
  std::string fullPath(R_ExpandFileName(path.c_str()));

  {
    auto st = TableWriter::OpenFile(fullPath, &table);
    if (!st.ok())
      stop("Failed to open '%s' (%s)", path, st.CodeAsString());
  }

  table->SetNumRows(df.nrows());
  CharacterVector names = df.names();

  for(int i = 0; i < df.size(); ++i) {
    table->AppendPlain(std::string(names[i]), toPrimitiveArray(df[i]));
  }

  {
    auto st = table->Finalize();
    if (!st.ok())
      stop("Failed to close '%s' (%s)", path, st.CodeAsString());
  }
}
