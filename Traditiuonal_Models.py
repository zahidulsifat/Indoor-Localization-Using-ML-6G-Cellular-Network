from math import log10, log, pi
from enum import Enum

class CityKind(Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2

class AreaType(Enum):
    OPEN = 0
    SUBURBAN = 1
    URBAN = 2

class TerrainKind(Enum):
    A = 0
    B = 1
    C = 2

class LeeAreaType(Enum):
    FreeSpace = (2.0, 45.0)
    OpenArea = (4.35, 49.0)
    SubUrban = (3.84, 61.7)
    Philadelphia = (3.68, 70.0)
    Newark = (4.31, 64.0)
    Tokyo = (3.05, 84.0)
    NewYorkCity = (4.80, 77.0)

class FreeSpace: #ok

    def __init__(self, freq = 800):

        self.freq = freq

    # end

    def pathLoss(self, dist):

        return 32.44 + 20*log10(dist) + 20*log10(self.freq)
        
    # end

# end


class OkumuraHata:

    def __init__(self, freq = 800, txH = 90, rxH = 1.2, cityKind = CityKind.MEDIUM, areaKind = AreaType.URBAN, checkFreqRange = False):

        self.freq = freq
        self.txH = txH
        self.rxH = rxH
        self.cityKind = cityKind
        self.areaKind = areaKind
        self.checkFreqRange = checkFreqRange

        self.invalidFreqRange = (freq < 500) or (freq > 1500)

    # end

    def pathLoss(self, dist):

        if self.checkFreqRange and self.invalidFreqRange:
            raise ValueError("OkumuraHata frequency range is 500MHz~1500MHz")
        # end

        if (self.freq <= 200) and (self.cityKind == 'large'):
            a = 8.29*(log10(1.54*self.rxH))^2 - 1.1
        elif (self.freq >= 400) and (self.cityKind == 'large'):
            a = 3.2*((log10(11.75*self.rxH)^2)) - 4.97
        else:
            a = (1.1*log10(self.freq-0.7))*self.rxH - (1.56*log10(self.freq-0.8))

        loss = 69.55 + (26.16)*log10(self.freq)-13.82*log10(self.txH) - a + (44.9-6.55*log10(self.txH))*log10(dist)

        if (self.areaKind is AreaType.OPEN):
            loss -= 4.78*((log10(self.freq))^2) + 18.33*log10(self.freq) - 40.94
        elif (self.areaKind is AreaType.SUBURBAN):
            loss -= 2*(log10(self.freq/28.0))^2 - 5.4

        return loss

    # end

# end

class Cost231Hata:

    def __init__(self, freq = 1650, txH = 50.0, rxH = 1.5, areaKind = AreaType.SUBURBAN, checkFreqRange = False):

        self.freq = freq
        self.txH = txH
        self.rxH = rxH
        self.areaKind = areaKind
        self.checkFreqRange = checkFreqRange

        self.invalidFreqRange = (freq < 1500) or (freq > 2000)

    # end

    def pathLoss(self, dist):

        if self.checkFreqRange and self.invalidFreqRange:
            raise ValueError("Cost231Hata frequency range is 1500MHz~2000MHz")
        # end

        ar = (1.1*log10(self.freq)-0.7)*self.rxH - (1.56*log(self.freq)-0.8)
        loss = 46.3 + 33.9*log10(self.freq) - 13.82*log10(self.txH) - ar + (44.9-6.55*log(self.txH))*log(dist)

        if (self.areaKind is AreaType.URBAN):
            loss += 3
        # end

        return loss

    # end

# end

class Cost231:

    def __init__(self, freq = 800, txH = 50.0, rxH = 1.5, ws = 20, bs = 10, hr = 35, cityKind = CityKind.SMALL, checkFreqRange = False):

        self.freq = freq
        self.txH = txH
        self.rxH = rxH
        self.ws = ws
        self.bs = bs
        self.hr = hr
        self.cityKind = cityKind
        self.checkFreqRange = checkFreqRange

        self.invalidFreqRange = (freq < 150) or (freq > 2000)

    # end

    def pathLoss(self, dist):

        deltaH = float(self.rxH) / float(self.txH)
        Lbsh = 0 if (self.hr > self.txH) else 18*log(1+deltaH)
        Ka = float(54)
        Kd = float(18)
        Kf = float(4)

        if (self.txH <= self.hr) and (dist >= 0.5):
            Ka -= 0.8*deltaH if (dist >= 0.5) else 0.8*deltaH*(dist/0.5)
        # end

        if (self.txH < self.hr):
            Kd -= 15*(self.txH-self.hr)/(self.hr-self.rxH)
        # end

        Kf += 0.7*(self.freq/925-1) if (self.cityKind is CityKind.SMALL) else 1.5*(self.freq/925-1)

        Lo = 32.4+20*log10(dist)+20*log10(self.freq)
        Lrts = 8.2+10*log(self.ws) + 10*log10(self.freq) + 10*log(deltaH)
        Lmsd = Lbsh + Ka + Kd*log10(dist) + Kf*log(self.freq) - 9*log10(self.bs)

        return Lo + Lrts + Lmsd

    # end

# end

class ECC33:

    def __init__ (self, freq = 800, txH = 50, rxH = 1.5, checkFreqRange = False):
        self.freq = freq
        self.txH = txH
        self.rxH = rxH
        self.checkFreqRange = checkFreqRange

    def pathLoss (self, dist):
        f = self.freq

        if self.checkFreqRange:
            if (f < 900 or f > 1900):
                raise ValueError('The frequency range for Ecc-33 Model is 900MHz-1900Mhz.')

        hb = self.txH
        hm = self.rxH   
        PLfs = 92.4+20*log10(dist)+20*log10(f/1000)
        PLbm = 20.41+9.83*log10(dist)+7.894*(log10(f/1000))+9.56*(log10(f/1000)) ** 2
        Gb = log10(hb/200)*(13.98+5.8*(log10(dist)) ** 2)
        Gm =(42.57+13.7*log10(f/1000))*(log10(hm)-0.585)
        PL= PLfs+PLbm-Gb-Gm

        return PL

class Ericsson:

    def __init__ (self, freq = 800, txH = 50, rxH = 1.5, cityKind = CityKind.SMALL, checkFreqRange = False):
        self.freq = freq
        self.txH = txH
        self.rxH = rxH
        self.cityKind = cityKind
        self.checkFreqRange = checkFreqRange


    def pathLoss (self, dist):
        f = self.freq
        hm = float(self.rxH)
        hb = float(self.txH)

        if self.checkFreqRange:
            if (f < 100 or f > 2000):
                raise ValueError('The frequency range for Ericcson Model is 500MHz-1500Mhz.')

        g = 44.49*log10(f)-4.78*(log10(f)) ** 2
        a2= 12
        a3= 0.1

        if self.cityKind is CityKind.SMALL:
            a0 = 36.2
            a1 = 30.2;
        elif self.cityKind is CityKind.MEDIUM:
            a0 = 43.2
            a1 = 68.9
        else:
            a0 = 45.9
            a1 = 100.6

        PL=a0+a1*log10(dist)+a2*log10(hb)+a3*(log10(hb))*(log10(dist))-3.2*log10((11.75*hm) ** 2)+g

        return PL

class Lee:
    def __init__ (self, freq = 900, txH = 50, rxH = 1.5, leeArea = LeeAreaType.SubUrban):
        self.freq = freq
        self.txH = txH
        self.rxH = rxH
        self.leeArea = leeArea

    def pathLoss (self, dist):
        hm = float(self.rxH)
        hb = float(self.txH)
        d = float(dist)
        f = float(self.freq)
        no = float(self.leeArea.value[0])
        po = float(self.leeArea.value[1])
        nf = float(3 if (f >= 850.0) else 2)
        L1 = -20 *log10(hb/30)
        X2 = 2 if (hm >= 3) else 1
        L2 = -10*X2*log10(hm/3)
        Lo = 50.3 + po - 10*no*log10(1.61)-10*nf*log10(900)
        L = Lo + 10*no*log10(d)+10*nf*log10(f)+L1+L2

        return L

class Sui:
    def __init__ (self, freq = 2500, txH = 50, rxH = 1.5, terrainKind = TerrainKind.A, shadowFading = 8.2, checkFreqRange = False):
        self.freq = freq
        self.txH = txH
        self.rxH = rxH
        self.terrainKind = terrainKind
        self.shadowFading = shadowFading
        self.checkFreqRange = checkFreqRange

    def pathLoss(self, dist):
        d = float(dist * 1000)
        f = self.freq

        if self.checkFreqRange:
            if (f < 1900 or f > 11000):
                raise ValueError('The SUI model frequency range is  1.9-11GHz\n')

        txH = float(self.txH)
        rxH = float(self.rxH)
        mode = self.terrainKind

        a = float(4.6)
        b = float(0.0075)
        c = float(12.6)
        s = float(self.shadowFading)  # shadow fading 8.2 dB and 10.6
        XhCF = float(-10.8)

        if mode is TerrainKind.B:
            a = 4.0
            b = 0.0065
            c = 17.1

        if mode is TerrainKind.C:
            a = 3.6
            b = 0.005
            c = 20
                #// shadow fading 8.2 dB and 10.6c
            XhCF = -20

        d0 = float(100)
        A = float(20 * log10((4 * pi * d0) / (300 / f)))
        y = float((a - b * txH) + c / txH)
        Xf = float(6 * log10(f / 2000))
        Xh = float(XhCF * log10(rxH / 2))

        return A + (10 * y * log10(d / d0)) + Xf + Xh + s

