#include <Rcpp.h>
using namespace Rcpp;

#include "feather/api.h"
using namespace feather;

#include "feather-types.h"
#include "feather-utils.h"


std::shared_ptr<OwnedMutableBuffer> makeBoolBuffer(int n) {
  int nbytes = util::bytes_for_bits(n);

  auto buffer = std::make_shared<OwnedMutableBuffer>();
  stopOnFailure(buffer->Resize(nbytes));
  util::fill_buffer(buffer->mutable_data(), 0, nbytes);

  return buffer;
}

// Primitive arrays ------------------------------------------------------------

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
    } else {
      // Valid
      util::set_bit(nulls, i);
      if (px[i]) {
        util::set_bit(values, i);
      }
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

static int set_null_bitmap(int* values, int length, uint8_t* nulls) {
  int n_missing = 0;
  for (int i = 0; i < length; ++i) {
    if (values[i] == NA_INTEGER) {
      ++n_missing;
    } else {
      // Valid
      util::set_bit(nulls, i);
    }
  }
  return n_missing;
}

// Returns an object with pointers into x, so must not be
// used after x goes away
PrimitiveArray intToPrimitiveArray(SEXP x) {
  int n = Rf_length(x);

  auto null_buffer = makeBoolBuffer(n);
  uint8_t* nulls = null_buffer->mutable_data();
  int n_missing = set_null_bitmap(INTEGER(x), n, nulls);

  PrimitiveArray out;
  out.type = PrimitiveType::INT32;
  out.length = n;
  out.values = reinterpret_cast<uint8_t*>(INTEGER(x));

  out.null_count = n_missing;
  if (n_missing > 0) {
    out.buffers.push_back(null_buffer);
    out.nulls = nulls;
  }

  return out;
}

PrimitiveArray factorCodesToPrimitiveArray(SEXP x) {
  // Must subtract 1 from factor codes for feather storage
  int n = Rf_length(x);

  auto null_buffer = makeBoolBuffer(n);
  auto nulls = null_buffer->mutable_data();
  int n_missing = set_null_bitmap(INTEGER(x), n, nulls);

  auto values_buffer = std::make_shared<OwnedMutableBuffer>();
  stopOnFailure(values_buffer->Resize(n * sizeof(int32_t)));
  util::fill_buffer(values_buffer->mutable_data(), 0, n);
  auto values = reinterpret_cast<int32_t*>(values_buffer->mutable_data());

  for (int i = 0; i < n; ++i) {
    int value = INTEGER(x)[i];
    // The null bits are set above
    if (value != NA_INTEGER)
      values[i] = value - 1;
  }

  PrimitiveArray out;
  out.type = PrimitiveType::INT32;
  out.length = n;
  out.values = reinterpret_cast<uint8_t*>(values);

  out.buffers.push_back(values_buffer);
  out.null_count = n_missing;
  if (n_missing > 0) {
    out.buffers.push_back(null_buffer);
    out.nulls = nulls;
  }

  return out;
}

PrimitiveArray rescaleToInt64(SEXP x, int64_t scale) {
  int n = Rf_length(x);

  auto null_buffer = makeBoolBuffer(n);
  auto nulls = null_buffer->mutable_data();

  auto values_buffer = std::make_shared<OwnedMutableBuffer>();
  stopOnFailure(values_buffer->Resize(n * sizeof(int64_t) / sizeof(int8_t)));
  util::fill_buffer(values_buffer->mutable_data(), 0, n);
  auto values = reinterpret_cast<int64_t*>(values_buffer->mutable_data());

  uint32_t n_missing = 0;
  switch(TYPEOF(x)) {
  case INTSXP: {
    int* px = INTEGER(x);
    for (int i = 0; i < n; ++i) {
      if (px[i] == NA_INTEGER) {
        ++n_missing;
      } else {
        // Valid
        util::set_bit(nulls, i);
        values[i] = px[i] * scale;
      }
    }
    break;
  }
  case REALSXP: {
    double* px = REAL(x);
    for (int i = 0; i < n; ++i) {
      if (R_IsNA(px[i])) {
        ++n_missing;
      } else {
        // Valid
        util::set_bit(nulls, i);
        values[i] = round(px[i] * scale);
      }
    }
    break;
  }
  default: stop("Unsupported type");
  }

  PrimitiveArray out;
  out.type = PrimitiveType::INT64;
  out.length = n;

  out.buffers.push_back(values_buffer);
  out.values = values_buffer->data();

  out.null_count = n_missing;
  if (n_missing > 0) {
    out.buffers.push_back(null_buffer);
    out.nulls = nulls;
  }

  return out;
}

PrimitiveArray dblToPrimitiveArray(SEXP x) {
  int n = Rf_length(x);

  auto null_buffer = makeBoolBuffer(n);
  auto nulls = null_buffer->mutable_data();
  uint32_t n_missing = 0;
  double* px = REAL(x);
  for (int i = 0; i < n; ++i) {
    if (R_IsNA(px[i])) {
      ++n_missing;
    } else {
      // Valid
      util::set_bit(nulls, i);
    }
  }

  PrimitiveArray out;
  out.type = PrimitiveType::DOUBLE;
  out.length = n;
  out.null_count = n_missing;
  out.values = reinterpret_cast<uint8_t*>(REAL(x));

  if (n_missing > 0) {
    out.buffers.push_back(null_buffer);
    out.nulls = nulls;
  }

  return out;
}

PrimitiveArray chrToPrimitiveArray(SEXP x) {
  int n = Rf_length(x), n_missing = 0;

  BufferBuilder data_builder;

  auto offsets_buffer = std::make_shared<OwnedMutableBuffer>();
  stopOnFailure(offsets_buffer->Resize(sizeof(int32_t) * (n + 1)));
  int32_t* offsets = reinterpret_cast<int32_t*>(offsets_buffer->mutable_data());
  int offset = 0, length = 0;

  auto null_buffer = makeBoolBuffer(n);
  auto nulls = null_buffer->mutable_data();

  for (int i = 0; i < n; ++i) {
    SEXP xi = STRING_ELT(x, i);

    if (xi == NA_STRING) {
      length = 0;
      ++n_missing;
    } else {
      // Valid
      util::set_bit(nulls, i);
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

// Columns ---------------------------------------------------------------------

Status addPrimitiveColumn(std::unique_ptr<TableWriter>& table,
                          const std::string& name, SEXP x) {
  switch(TYPEOF(x)) {
  case LGLSXP:
    return table->AppendPlain(name, lglToPrimitiveArray(x));
  case INTSXP:
    return table->AppendPlain(name, intToPrimitiveArray(x));
  case REALSXP:
    return table->AppendPlain(name, dblToPrimitiveArray(x));
  case STRSXP:
    return table->AppendPlain(name, chrToPrimitiveArray(x));
  default:
    std::string msg = tfm::format("%s is a %s", name, Rf_type2char(TYPEOF(x)));
    return Status::NotImplemented(msg);
  }
}

Status addCategoryColumn(std::unique_ptr<TableWriter>& table,
                         const std::string& name, SEXP x) {
  if (TYPEOF(x) != INTSXP)
    stop("'%s' is corrupt", name);

  SEXP x_levels = Rf_getAttrib(x, Rf_install("levels"));
  if (TYPEOF(x_levels) != STRSXP)
    stop("'%s' is corrupt", name);

  auto values = factorCodesToPrimitiveArray(x);
  auto levels = chrToPrimitiveArray(x_levels);
  bool ordered = Rf_inherits(x, "ordered");

  return table->AppendCategory(name, values, levels, ordered);
}

Status addDateColumn(std::unique_ptr<TableWriter>& table,
                     const std::string& name, SEXP x) {
  // dates can be stored either as integers or doubles
  if (TYPEOF(x) != INTSXP && TYPEOF(x) != REALSXP)
    stop("%s is corrupt", name);
  auto values = intToPrimitiveArray(as<IntegerVector>(x));

  return table->AppendDate(name, values);
}

Status addTimeColumn(std::unique_ptr<TableWriter>& table,
                     const std::string& name, SEXP x) {
  if (TYPEOF(x) != INTSXP && TYPEOF(x) != REALSXP)
    stop("%s is corrupt", name);
  auto values = rescaleToInt64(x, 1e6);

  TimeMetadata metadata;
  metadata.unit = TimeUnit::MICROSECOND;

  return table->AppendTime(name, values, metadata);
}


Status addTimestampColumn(std::unique_ptr<TableWriter>& table,
                          const std::string& name, SEXP x) {
  if (TYPEOF(x) != INTSXP && TYPEOF(x) != REALSXP)
    stop("%s is corrupt", name);
  auto values = rescaleToInt64(x, 1e6);

  SEXP tzoneR = Rf_getAttrib(x, Rf_install("tzone"));
  std::string tzone = Rf_isNull(tzoneR) ? "UTC" : Rf_translateCharUTF8(STRING_ELT(tzoneR, 0));

  TimestampMetadata metadata;
  metadata.unit = TimeUnit::MICROSECOND;
  metadata.timezone = tzone;

  return table->AppendTimestamp(name, values, metadata);
}


Status addColumn(std::unique_ptr<TableWriter>& table,
                 const std::string& name, SEXP x) {
  if (Rf_inherits(x, "factor")) {
    return addCategoryColumn(table, name, x);
  } else if (Rf_inherits(x, "Date")) {
    return addDateColumn(table, name, x);
  } else if (Rf_inherits(x, "time") || Rf_inherits(x, "hms")) {
    return addTimeColumn(table, name, x);
  } else if (Rf_inherits(x, "POSIXct")) {
    return addTimestampColumn(table, name, x);
  } else if (Rf_inherits(x, "POSIXlt")) {
    stop("Can not write POSIXlt (%s). Convert to POSIXct first.", name);
    return Status::NotImplemented("");
  } else {
    return addPrimitiveColumn(table, name, x);
  }
}

// [[Rcpp::export]]
void writeFeather(DataFrame df, const std::string& path) {
  std::unique_ptr<TableWriter> table;
  std::string fullPath(R_ExpandFileName(path.c_str()));

  stopOnFailure(TableWriter::OpenFile(fullPath, &table));

  table->SetNumRows(df.nrows());
  CharacterVector names = df.names();

  for(int i = 0; i < df.size(); ++i) {
    stopOnFailure(addColumn(table, std::string(names[i]), df[i]));
  }

  stopOnFailure(table->Finalize());
}
