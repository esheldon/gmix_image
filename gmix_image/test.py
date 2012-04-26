import _gmix_image

def dotest():
    gd = [{'p':0.4,'row':35,'col':66,'irr':2.5,'irc':0.1,'icc':3.1},
          {'p':0.3,'row':22,'col':55,'irr':1.7,'irc':0.3,'icc':1.5}]
    gv = _gmix_image.GVec(gd)
    gv.print_n()
    print gv

    gm = _gmix_image.GMix(gv)
    gm.print_n()
    print gm
if __name__ == "__main__":
    dotest()
