// Copyright Carlos Scheidegger, 2016
// License: Apache 2.0 (same as Feather itself)
//
// A minimal JavaScript implementation of feather file reading. Notably,
// this only currently supports plain encodings, doesn't support null values,
// and doesn't support int64, uint64, or boolean data.
//
// This uses https://github.com/google/flatbuffers/blob/master/js/flatbuffers.js
// and metadata_generated.js
//
// if metadata.fbs changes, you will need to regenerate metadata_generated.js,
// by
//
//   $ flatc --js metadata.fbs
//
//////////////////////////////////////////////////////////////////////////////

// If you want to bowerize/browserify/complexify this thing, go right ahead.

// Extra function to the UTF-8 Case
/* utf.js - UTF-8 <=> UTF-16 convertion
 * Copyright (C) 1999 Masanao Izumo <iz@onicos.co.jp>
 */

function Utf8ArrayToStr(array) {
    var out, i, len, c;
    var char2, char3;

    out = "";
    len = array.length;
    i = 0;
    while(i < len) {
    c = array[i++];
    switch(c >> 4)
    {
      case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7:
        // 0xxxxxxx
        out += String.fromCharCode(c);
        break;
      case 12: case 13:
        // 110x xxxx   10xx xxxx
        char2 = array[i++];
        out += String.fromCharCode(((c & 0x1F) << 6) | (char2 & 0x3F));
        break;
      case 14:
        // 1110 xxxx  10xx xxxx  10xx xxxx
        char2 = array[i++];
        char3 = array[i++];
        out += String.fromCharCode(((c & 0x0F) << 12) |
                       ((char2 & 0x3F) << 6) |
                       ((char3 & 0x3F) << 0));
        break;
    }
    }

    return out;
}


(function(module) {
    var typeEnum = {
        0:  Uint32Array,//Boolean
        1:  Int8Array,
        2:  Int16Array,
        3:  Int32Array,
        5:  Uint8Array,
        6:  Uint16Array,
        7:  Uint32Array,
        9:  Float32Array,
        10: Float64Array,
        11: Uint32Array //UTF8
    };

    var sizeEnum = {
        0:  4,
        1:  1,
        2:  2,
        3:  4,
        5:  1,
        6:  2,
        7:  4,
        9:  2,
        10: 4,
        11: 4
    };

    /**
     * convert a feather file loaded as an ArrayBuffer to an object whose
     * slots are the columns of the feather file, and whose values are Typed Arrays
     * which share the buffer.
     *
     * This should be about as efficient as we can get in JavaScript.
     *
     * NB: This means that if you mutate the buffer, the values in the
     * typed arrays will change (and vice versa!)
     *
     * @param {ArrayBuffer} buffer The buffer containing the feather file to convert.
     */
    function convertToDataFrame(buffer)
    {
        var FEA1 = 826361158;
        var l = buffer.byteLength;
        if (l < 8) {
            throw new Error("Feather file needs to be at least 8 bytes, got " + l + " instead.");
        }
        var byteView = new Uint8Array(buffer, 0, l);
        var intView = new Uint32Array(buffer, 0, l/4);
        var beginMagic = intView[0];
        var endMagic = intView[intView.length-1];

        if (beginMagic !== FEA1) {
            var str = String.fromCharCode(byteView[0]) +
                    String.fromCharCode(byteView[1]) +
                    String.fromCharCode(byteView[2]) +
                    String.fromCharCode(byteView[3]);
            throw new Error("Expected magic number at BOF to be FEA1, got " + str + " instead.");
        }

        if (endMagic !== FEA1) {
            var str = String.fromCharCode(byteView[l-4]) +
                    String.fromCharCode(byteView[l-3]) +
                    String.fromCharCode(byteView[l-2]) +
                    String.fromCharCode(byteView[l-1]);
            throw new Error("Expected magic number at EOF to be FEA1, got " + str + " instead.");
        }

        var metadataLength = intView[intView.length-2];

        var metadata = buffer.slice(buffer.byteLength-8-metadataLength, buffer.byteLength-8);

        var bb = new flatbuffers.ByteBuffer(new Uint8Array(metadata));
        var r = feather.fbs.CTable.getRootAsCTable(bb);
        var nCols = r.columnsLength();

        var d = {};
        for (var i=0; i<nCols; ++i) {
            var c = r.columns(i);
            var name = c.name();
            var values = c.values();
            if (values.encoding() !== 0) {
                throw new Error("Only PLAIN encoding is currently supported.");
            }
            var offset = values.offset().toFloat64();
            var nullCount = values.nullCount().toFloat64();
            var nElements = values.length().toFloat64();
            var type = values.type();

            if (typeEnum[type] === undefined) {
                throw new Error("Unsupported datatype:"+type);
            }
            // Special case for Boolean
            // Each bool is a bit inside the array
            if(type === 0){
                var null_set = [];
                // Get the null block and the new offset if there are nulls
                if(nullCount > 0){
                    null_set = new Uint8Array(buffer, offset, Math.ceil(nElements/8));
                    var n_min_bytes = Math.ceil(nElements/8);
                    // Padding to 8 bits
                    offset = offset + Math.ceil(n_min_bytes/8)*8;
                }

                // The bool data
                var bool_data = new Uint8Array(buffer, offset, Math.ceil(nElements/8));

                d[name] = [];

                // For each element we need to get its bit
                for(var element = 0; element < nElements; element++){
                    // Check if its a null value
                    if(nullCount>0 && !(null_set[Math.floor(element/8)] & ( 1 << (element % 8) ) ) ){
                        d[name][element] = null;
                    }else{
                        d[name][element] = ( bool_data[Math.floor(element/8)] & ( 1 << (element % 8) ) ) > 0 ? true : false;
                    }
                }
            // Special case for UTF-8
            // There is a additional buffer for each string size
            }else if(type === 11){
                var null_set = [];
                // Get the null block and the new offset if there are nulls
                if(nullCount > 0){
                    null_set = new Uint8Array(buffer, offset, Math.ceil(nElements/8));
                    var n_min_bytes = Math.ceil(nElements/8);
                    // Padding to 8 bits
                    offset = offset + Math.ceil(n_min_bytes/8)*8;
                }
                // Special buffer for UTF-8
                var sizes_utf = new Uint32Array(buffer, offset, nElements+1);
                // Consider in the offset
                var pad = (Math.ceil(((nElements+1)*4)/8))*8;

                d[name] = [];

                for(var element=0; element < nElements; element++){
                    // Check if its a null value
                    if(nullCount>0 && !(null_set[Math.floor(element/8)] & ( 1 << (element % 8) ) ) ){
                        d[name][element] = null;
                    }else{
                        var s = new Uint8Array(buffer, offset+pad+sizes_utf[element], sizes_utf[element+1]-sizes_utf[element]);
                        d[name][element] =  Utf8ArrayToStr(s);
                    }
                }

              }else{
                  var null_set = [];
                  // Get the null block and the new offset if there are nulls
                  if(nullCount>0){
                      null_set = new Uint8Array(buffer, offset, Math.ceil(nElements/8));
                      n_min_bytes = Math.ceil(nElements/8);
                      offset = offset + Math.ceil(n_min_bytes/8)*8;
                  }

                  d[name] = [];
                  var variables = new typeEnum[type](buffer, offset, nElements);
                  for(var element=0; element < nElements; element++){
                      // Check if its a null value
                      if(nullCount>0 && !(null_set[Math.floor(element/8)] & ( 1 << (element % 8) ) ) ){
                          d[name][element] = null;
                      }else{
                          d[name][element] = variables[element];
                      }
                  }
              }

        }
        return d;
    }

    module.convertToDataFrame = convertToDataFrame;
}(window));
