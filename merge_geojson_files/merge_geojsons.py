from json import load, JSONEncoder
from optparse import OptionParser
from re import compile
import os
import glob
import geojson

float_pat = compile(r'^-?\d+\.\d+(e-?\d+)?$')
charfloat_pat = compile(r'^[\[,\,]-?\d+\.\d+(e-?\d+)?$')

parser = OptionParser(usage="""%prog [options]
Group multiple GeoJSON files into one output file.
Example:
  python %prog -p 2 input-1.json input-2.json output.json""")

defaults = dict(precision=6)

parser.set_defaults(**defaults)

parser.add_option('-p', '--precision', dest='precision',
                  type='int', help='Digits of precision, default %(precision)d.' % defaults)

if __name__ == '__main__':

    json_dir_name = r"D:\LUMS_RA\Data\WHU_Data\Test\Masks"
    json_pattern = os.path.join(json_dir_name, '*.geojson')
    file_list = glob.glob(json_pattern)

    outjson = dict(type='FeatureCollection', features=[])

    for infile in file_list:
        injson = load(open(infile))

        if injson.get('type', None) != 'FeatureCollection':
            raise Exception('Sorry, "%s" does not look like GeoJSON' % infile)

        if type(injson.get('features', None)) != list:
            raise Exception('Sorry, "%s" does not look like GeoJSON' % infile)

        outjson['features'] += injson['features']

    encoder = JSONEncoder(separators=(',', ':'))
    encoded = encoder.iterencode(outjson)

    # format = '%.' + str(options.precision) + 'f'
    output = open(r'D:\LUMS_RA\Data\WHU_Data\Test\merged_test.geojson', 'w')

    for token in encoded:
        # if charfloat_pat.match(token):
        #     # in python 2.7, we see a character followed by a float literal
        #     output.write(token[0] + format % float(token[1:]))
        #
        # elif float_pat.match(token):
        #     # in python 2.6, we see a simple float literal
        #     output.write(format % float(token))
        #
        # else:
        output.write(token)