from random import getrandbits


def main():
    bits = getrandbits(256)
    print("%032x" % bits)


if __name__ == '__main__':
    main()
