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

PrimitiveArray lglToPrimitiveArray(SEXP x) {
  int n = Rf_length(x), n_missing = 0;

  auto null_buffer = makeBoolBuffer(n),
       values_buffer = makeBoolBuffer(n);
  auto nulls = null_buffer->mutable_data(),
       values = values_buffer->mutable_data();

  int* px = INTEGER(x);
  for (int i = 0; i < n; ++i) {
    if (px[i] == NA_LOGICAL) {
      ++n_missing;
      util::set_bit(nulls, i);
    } else if (px[i]) {
      util::set_bit(values, i);
    }
  }

  PrimitiveArray out;
  out.type = PrimitiveType::BOOL;
  out.length = n;

  out.buffers.push_back(values_buffer);
  out.values = values;

  if (n_missing > 0) {
    out.null_count = n_missing;
    out.buffers.push_back(null_buffer);
    out.nulls = nulls;
  } else {
    out.null_count = 0;
  }

  return out;

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

PrimitiveArray chrToPrimitiveArray(SEXP x) {
  int n = Rf_length(x), n_missing = 0;

  BufferBuilder data_builder;

  auto offsets_buffer = std::make_shared<OwnedMutableBuffer>();
  offsets_buffer->Resize(sizeof(int32_t) * (n + 1));
  int32_t* offsets = reinterpret_cast<int32_t*>(offsets_buffer->mutable_data());
  int offset = 0, length = 0;

  auto null_buffer = makeBoolBuffer(n);
  auto nulls = null_buffer->mutable_data();

  for (int i = 0; i < n; ++i) {
    SEXP xi = STRING_ELT(x, i);

    if (xi == NA_STRING) {
      util::set_bit(nulls, i);
      ++n_missing;
    } else {
      const char* utf8 = Rf_translateCharUTF8(xi);
      length = strlen(utf8);
      data_builder.Append(reinterpret_cast<const uint8_t*>(utf8), length);
    }

    offsets[i] = offset;
    offset += length;
  }
  offsets[n] = offset;

  PrimitiveArray out;
  out.type = PrimitiveType::UTF8;
  out.length = n;

  std::shared_ptr<Buffer> data_buffer = data_builder.Finish();
  out.values = data_buffer->data();
  out.buffers.push_back(data_buffer);

  out.offsets = offsets;
  out.buffers.push_back(offsets_buffer);

  out.null_count = n_missing;
  if (n_missing > 0){
    out.nulls = nulls;
    out.buffers.push_back(null_buffer);
  }

  return out;
}


void addRColumn(std::unique_ptr<TableWriter>& table, std::string name, SEXP x) {
  switch(TYPEOF(x)) {
  case LGLSXP:  table->AppendPlain(name, lglToPrimitiveArray(x));
  case INTSXP:  table->AppendPlain(name, intToPrimitiveArray(x));
  case REALSXP: table->AppendPlain(name, dblToPrimitiveArray(x));
  case STRSXP:  table->AppendPlain(name, chrToPrimitiveArray(x));
  default:
    stop("Unsupported type (%s)", Rf_type2char(TYPEOF(x)));
    throw 0;
  }
}

// [[Rcpp::export]]
void writeFeather(DataFrame df, std::string path) {
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
    addRColumn(table, std::string(names[i]), df[i]);
  }

  {
    auto st = table->Finalize();
    if (!st.ok())
      stop("Failed to close '%s' (%s)", path, st.CodeAsString());
  }
}
