import os
import subprocess
import uuid

from feather.compat import guid, tobytes

R_READ_WRITE_TEMPLATE = """
library(feather)

df <- read_feather("{0}")
write_feather(df, "{1}")
"""


def random_path():
    return 'feather_{}'.format(guid())


def remove_paths(paths):
    for x in paths:
        try:
            os.remove(x)
        except os.error:
            pass


def run_rcode(code):
    tmp_r_path = 'test_{0}.R'.format(uuid.uuid4().hex)

    with open(tmp_r_path, 'wb') as f:
        f.write(tobytes(code))

    cmd = ['Rscript', tmp_r_path]
    try:
        subprocess.check_output(cmd)
    finally:
        remove_paths([tmp_r_path])


def roundtrip_r(input_path, output_path):
    code = R_READ_WRITE_TEMPLATE.format(input_path, output_path)
    run_rcode(code)
