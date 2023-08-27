from inspect import currentframe, getframeinfo

print('line 2 in danutils')
def filename_n_line():
    cf = currentframe()
    #DAN: f_back is the function that called this
    filename = getframeinfo(cf.f_back).filename
    linum = cf.f_back.f_lineno
    print(filename, linum)


def filename_n_line_str():
    cf = currentframe()
    #DAN: f_back is the function that called this
    filename = getframeinfo(cf.f_back).filename
    linum = cf.f_back.f_lineno
    return filename + ':' + str(linum) + '    '


def printd(msg):
    cf = currentframe()
    #DAN: f_back is the function that called this
    filename = getframeinfo(cf.f_back).filename
    linum = cf.f_back.f_lineno
    print(filename + ':' + str(linum) + msg)


def bp():
    cf = currentframe()
    #DAN: f_back is the function that called this
    filename = getframeinfo(cf.f_back).filename
    linum = cf.f_back.f_lineno
    print(filename, linum)
    while True:
        inpt = input(f'debug stuff \nVVVVVVVVV \n')
        if inpt == 'fin':
            break
        try:
            eval('print(' + inpt + ')')
        except Exception:
            print('Invalid Input: ' + inpt)
            continue