import sys

from test_annotations import test_dataset, FittingMode

if __name__ == '__main__':
    d1 = test_dataset(sys.argv[1], FittingMode.LastPoint, True)
    print("total with last point: {}".format(d1))
