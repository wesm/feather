import pandas
import scipy
import feather

import os
import glob

##############################################################################

feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1.0])}), "1.feather")
feather.write_dataframe(pandas.DataFrame({"b": scipy.array([1.0])}), "2.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([2.0])}), "3.feather")
feather.write_dataframe(pandas.DataFrame({"b": scipy.array([2.0])}), "4.feather")

feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1.0, 2.0])}), "5.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1.0, 2.0, 3.0, 4.0])}), "6.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1.0, 2.0, 3.0, 4.0]),
                                          "b": scipy.array([5.0, 6.0, 7.0, 8.0])}), "7.feather")



feather.write_dataframe(pandas.DataFrame({"hello": scipy.array([1.0, 2.0, 3.0, 4.0])}), "8.feather")

feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1.0,  2.0,  3.0,  4.0]),
                                          "b": scipy.array([5.0,  6.0,  7.0,  8.0]),
                                          "c": scipy.array([9.0,  10.0, 11.0, 12.0])}), "9.feather")

feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1.0,  2.0,  3.0,  4.0]),
                                          "b": scipy.array([5.0,  6.0,  7.0,  8.0]),
                                          "c": scipy.array([9.0,  10.0, 11.0, 12.0]),
                                          "d": scipy.array([13.0, 14.0, 15.0, 16.0])}), "10.feather")

feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1, 2, 3, 4], dtype='int32'),
                                          "b": scipy.array([5.0, 6.0, 7.0, 8.0])}), "11.feather")

feather.write_dataframe(pandas.DataFrame({"a": scipy.array([0,0,0,0], dtype='float32')}), "float32.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([0,0], dtype='float64')}), "float64.feather")

feather.write_dataframe(pandas.DataFrame({"a": scipy.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype='int8')}), "int8.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([0,0,0,0,0,0,0,0], dtype='int16')}), "int16.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([0,0,0,0], dtype='int32')}), "int32.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([0,0], dtype='int64')}), "int64.feather")

feather.write_dataframe(pandas.DataFrame({"a": scipy.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype='uint8')}), "uint8.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([0,0,0,0,0,0,0,0], dtype='uint16')}), "uint16.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([0,0,0,0], dtype='uint32')}), "uint32.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([0,0], dtype='uint64')}), "uint64.feather")

for i in xrange(16):
    feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1] * i, dtype='int8')}), "int8_%sels.feather" % i)
    feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1] * i, dtype='uint8')}), "uint8_%sels.feather" % i)
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1] * 16, dtype='int8')}), "int8_16els.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1] * 32, dtype='int8')}), "int8_32els.feather")
feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1] * 48, dtype='int8')}), "int8_48els.feather")

for i in xrange(16):
    feather.write_dataframe(pandas.DataFrame({"a": scipy.array([1] * i, dtype='int8'),
                                              "b": scipy.array([2] * i, dtype='uint8')}), "int8_uint8_%sels.feather" % i)
    feather.write_dataframe(pandas.DataFrame({"b": scipy.array([2] * i, dtype='int8'),
                                              "c": scipy.array([1] * i, dtype='uint8'),
                                              }), "uint8_int8_%sels-1.feather" % i)
    feather.write_dataframe(pandas.DataFrame({"b": scipy.array([2] * i, dtype='int8'),
                                              "a": scipy.array([1] * i, dtype='uint8')}), "uint8_int8_%sels-2.feather" % i)

feather.write_dataframe(pandas.DataFrame({"b": scipy.array([1] * 3, dtype='int8'),
                                          }), "1_arrays.feather")

feather.write_dataframe(pandas.DataFrame({"b": scipy.array([1] * 3, dtype='int8'),
                                          "c": scipy.array([2] * 3, dtype='int8'),
                                          }), "2_arrays.feather")

feather.write_dataframe(pandas.DataFrame({"b": scipy.array([1] * 3, dtype='int8'),
                                          "c": scipy.array([2] * 3, dtype='int8'),
                                          "d": scipy.array([3] * 3, dtype='int8'),
                                          }), "3_arrays.feather")

feather.write_dataframe(pandas.DataFrame({"b": scipy.array([1] * 3, dtype='int8'),
                                          "c": scipy.array([2] * 3, dtype='int8'),
                                          "d": scipy.array([3] * 3, dtype='int8'),
                                          "e": scipy.array([4] * 3, dtype='int8'),
                                          }), "4_arrays.feather")

feather.write_dataframe(pandas.DataFrame({"b": scipy.array([1] * 3, dtype='int8'),
                                          "c": scipy.array([2] * 3, dtype='int8'),
                                          "d": scipy.array([3] * 3, dtype='int8'),
                                          "e": scipy.array([4] * 3, dtype='int8'),
                                          "f": scipy.array([5] * 3, dtype='int8'),
                                          }), "5_arrays.feather")

##############################################################################

for n in glob.glob("*.feather"):
    cmd = 'hexdump -e \'"%%07.7_ax  " 8/1 "%%02x "\' -e \'" |" 8/1 "%%_p" "|\n"\' %s > %s' % (n, n.replace('.feather', '.dump'))
    os.system(cmd)

# feather.write_dataframe("7.feather", pandas.DataFrame({"a": scipy.array([1.0]), "b": scipy.array([1.0])}))
# feather.write_dataframe("8.feather", pandas.DataFrame({"a": scipy.array([1.0]), "b": scipy.array([1.0])}))
