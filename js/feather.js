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

(function(module) {
    var typeEnum = {
        1:  Int8Array,
        2:  Int16Array,
        3:  Int32Array,
        5:  Uint8Array,
        6:  Uint16Array,
        7:  Uint32Array,
        9:  Float32Array,
        10: Float64Array
    };

    var sizeEnum = {
        1:  1,
        2:  2,
        3:  4,
        5:  1,
        6:  2,
        7:  4,
        9:  2,
        10: 4
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
            if (nullCount !== 0) {
                throw new Error("Null values not currently supported");
            }
            var nElements = values.length().toFloat64();
            var type = values.type();
            if (typeEnum[type] === undefined) {
                throw new Error("Unsupported datatype");
            }
            d[name] = new typeEnum[type](buffer, offset, nElements);
        }
        return d;
    }
    
    module.convertToDataFrame = convertToDataFrame;
}(window));
