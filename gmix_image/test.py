import _gmix_image

def dotest():
    gv = _gmix_image.GVec(3)
    gv.print_n()
    print gv

    gm = _gmix_image.GMix(gv)
    gm.print_n()
    print gm
if __name__ == "__main__":
    dotest()
