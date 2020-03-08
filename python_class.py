class Fraction:

    @staticmethod
    def info():
        print("this the class for Fraction")
    def __init__(self,nume, deno=1):
        if deno == 0:
            raise ZeroDivisionError()
        nume1, deno1 = self.__create_fraction(nume)
        nume2, deno2 = self.__create_fraction(deno)
        self.__nume, self.__deno = self.__redution(nume1 * deno2, deno1 * nume2)

    def __create_fraction(self, num):
        length = len(str(num).split('.')[-1]) if '.' in str(num) else 0
        nume = int(num * 10 ** length)
        deno = int(10 ** length)
        return self.__redution(nume, deno)

    def __add__(self,other):
        other = Fraction(other) if type(other) != type(self) else other
        new_nume = self.__nume * other.__deno + self.__deno * other.__nume
        new_deno = self.__deno * other.__deno
        res = Fraction(new_nume, new_deno)
        return res
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = Fraction(other) if type(other) != type(self) else other
        res = Fraction(self.__nume * other.__nume, self.__deno * other.__deno)
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __truediv__(self, other):
        other = Fraction(other) if type(other) != type(self) else other
        res = Fraction(self.__nume * other.__deno, self.__deno * other.__nume)
        return res

    def __rtruediv__(self, other):
        other = Fraction(other) if type(other) != type(self) else other
        res = Fraction( self.__deno * other.__nume, self.__nume * other.__deno)
        return res


    def __redution(self, nume, deno):
        # fractional reduction
        if nume % deno == 0:
            nume = nume // deno
            deno = 1
        else:
            a, b = nume, deno
            while a % b != 0:
                a, b = b, a % b
            nume = nume // b
            deno = deno // b
        return nume, deno

    def __pow__(self, power, modulo=None):
        return Fraction(self.__nume ** power, self.__deno ** power)

    def __repr__(self):
        if self.__deno != 1:
            return "{}/{}".format(self.__nume, self.__deno)
        else:
            return "{}".format(self.__nume)

    def __float__(self):
        return self.__nume / self.__deno

    def __int__(self):
        return int(self.__float__())

    def __neg__(self):
        return Fraction(-self.__nume, self.__deno)

    def __le__(self, other):
        # <=
        return self.__lt__(other) or self.__eq__(other)

    def __eq__(self, other):
        return (self - other).__nume == 0

    def __lt__(self, other):
        # <
        return (self - other).__nume < 0

    def __ge__(self, other):
        return self > other or self == other

    def __gt__(self, other):
        return (self - other).__nume > 0

    def __ne__(self, other):
        return not self == other

def twoSum(nums, target) :
    mydict = {}
    for index, value in enumerate(nums):
        if value in mydict:
            return [mydict[value], index]
        mydict[target - value] = index

 