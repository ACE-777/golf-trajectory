import sys

from test_annotations import test_dataset

if __name__ == '__main__':
    d1 = test_dataset(sys.argv[1], True, True)
    print("total with last point: {}".format(d1))
