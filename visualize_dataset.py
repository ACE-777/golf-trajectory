import sys

from test_annotations import test_dataset, FittingMode

if __name__ == '__main__':
    alg = 'SLSQP'
    if len(sys.argv) > 2:
        alg = sys.argv[2]
    d1 = test_dataset(sys.argv[1], FittingMode.ApexAndLast, True, alg, 5)
    print("total with last point: {}".format(d1))
