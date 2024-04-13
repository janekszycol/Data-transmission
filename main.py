import numpy as np
import matplotlib.pyplot as plt

W = 10
fs = 500
Tc = 1
N = Tc * fs
N = int(N)


def decimalToBinary(n):
    return bin(n).replace("0b", "")


def str2bits(s):
    b = []
    for i in range(len(s)):
        nb = ord(s[i])
        nbb = decimalToBinary(nb)
        for n in range(len(nbb)):
            b.append(int(nbb[n]))
    #print(b)
    return b


def modASK(B):
    A1 = 1
    A2 = 0.5
    Tb = Tc / len(B)
    fn = W / Tb
    zakres=N/len(B)
    zakres = int(zakres)

    z1=0
    z2=-zakres
    Za = []
    t=[]

    for i in range(0, N):
        t.append(i / fs)
    for b in B:
        z1 = z1+zakres
        z2 = z2+zakres
        for i in range(z2,z1):
            if b == 0:
                Za.append(A1 * np.sin(2 * np.pi * fn * t[i]))
            else:
                Za.append(A2 * np.sin(2 * np.pi * fn * t[i]))
    zm=N-len(Za)
    l = [0.0] * zm
    Za=Za+l
    return Za



def demodASK(z,B):
    t = []
    for i in range(0, N):
        t.append(i / fs)
    Tb = Tc / len(B)
    fn = W / Tb
    A = 1

    zakres = N / len(B)
    zakres = int(zakres)

    z1 = 0
    z2 = -zakres

    Z = []
    X = []
    Xlist = []


    for n in range(N):
        Z.append(A * np.sin(2 * np.pi * fn * t[n]))
    #print(Z)
    for n in range(N):
        X.append(z[n] * Z[n])



    # Sumlist=[]
    # for i in range(0, int(len(B))):
    #     sum = 0
    #     z1 =z1+zakres
    #     z2 = z2+zakres
    #     for j in range(z2, z1):
    #         sum = sum + X[j]
    #         Sumlist.append(sum)
    #     #print(sum)
    #     Xlist.append(sum)

    licz = N / len(B)
    j = 0
    Sumlist = []
    for n in range(len(B)):
        sum = 0
        for i in range(int(licz)):
            sum = sum + X[j + i]
            Sumlist.append(sum)
        Xlist.append(sum)
        j = j + int(licz)
    #h = np.average(Xlist)
    h=0.5
    # plt.plot(Xlist)
    # plt.show()
    Clist = []
    for n in range(len(B)):
        if Xlist[n] < h:
            Clist.append(1)
        else:
            Clist.append(0)


    return Clist

def modPSK(B):
    Tb = Tc / len(B)
    fn = W / Tb
    Zp = []
    zakres = N / len(B)
    zakres = int(zakres)
    # print('zakres')
    # print(zakres)

    z1 = 0
    z2 = -zakres
    t = []

    for i in range(0, N):
        t.append(i / fs)

    for b in B:
        z1=z1+zakres
        z2=z2+zakres
        for i in range(z2,z1):
            if b == 0:
                Zp.append(np.sin(2 * np.pi * fn * t[i]))
            else:
                Zp.append(np.sin(2 * np.pi * fn * t[i] + np.pi))

    zm = N - len(Zp)
    l = [0.0] * zm
    Zp = Zp + l
    return Zp


def demodPSK(Zpsk,B):
    t = []
    for i in range(0, N):
        t.append(i / fs)

    Z = []
    X = []
    Xlist = []
    Tb = Tc / len(B)
    fn = W / Tb
    A = 1


    for n in range(N):
        Z.append(A * np.sin(2 * np.pi * fn * t[n]))

    for n in range(N):
        X.append(Zpsk[n] * Z[n])

    licz = N / len(B)
    #print(len(B))
    j = 0
    Sumlist = []
    for n in range(len(B)):
        sum = 0
        for i in range(int(licz)):
            sum = sum + X[j + i]
            Sumlist.append(sum)
        Xlist.append(sum)
        j = j + int(licz)


    Clist = []
    for n in range(len(B)):
        if Xlist[n] < 0:
            Clist.append(1)
        else:
            Clist.append(0)

    return Clist


def modFSK(B):
    Tb = Tc / len(B)
    fn1 = (W + 1) / Tb
    fn2 = (W + 2) / Tb
    Zf = []
    zakres = N / len(B)
    zakres = int(zakres)
    # print('zakres')
    # print(zakres)

    z1 = 0
    z2 = -zakres
    t = []

    for i in range(0, N):
        t.append(i / fs)

    for b in B:
        z1=z1+zakres
        z2=z2+zakres
        for n in range(z2,z1):
            if b == 0:
                Zf.append(np.sin(2 * np.pi * fn1 * t[n]))
            else:
                Zf.append(np.sin(2 * np.pi * fn2 * t[n]))


    zm = N - len(Zf)
    l = [0.0] * zm
    Zf = Zf + l
    return Zf


def demodFSK(Zfsk,B):
    Z1 = []
    Z2 = []
    X1 = []
    X2 = []
    Xlist1 = []
    Xlist2 = []
    Tb = Tc / len(B)
    fn1 = (W + 1) / Tb
    fn2 = (W + 2) / Tb
    A=1
    t = []
    for i in range(0, N):
        t.append(i / fs)

    for n in range(N):
        Z1.append(A * np.sin(2 * np.pi * fn1 * t[n]))
    for n in range(N):
        X1.append(Zfsk[n] * Z1[n])

    for n in range(N):
        Z2.append(A * np.sin(2 * np.pi * fn2 * t[n]))
    for n in range(N):
        X2.append(Zfsk[n] * Z2[n])

    zakres = N / len(B)
    zakres = int(zakres)
    # print('zakres')
    # print(zakres)

    z1 = 0
    z2 = -zakres

    licz = N / len(B)
    j = 0
    Sumlist1 = []
    for n in range(len(B)):
        sum = 0
        for i in range(int(licz)):
            sum = sum + X1[j + i]
            Sumlist1.append(sum)
        Xlist1.append(sum)
        j = j + int(licz)

    #niezmieniona
    # Sumlist1 = []
    # for n in range(len(B)):
    #     sum = 0
    #     z1 = z1 + zakres
    #     z2 = z2 + zakres
    #     for j in range(z2, z1):
    #         sum = sum + X1[j]
    #         Sumlist1.append(sum)
    #     Xlist1.append(sum)


    z1 = 0
    z2 = -zakres

    licz = N / len(B)
    j = 0
    Sumlist2 = []
    for n in range(len(B)):
        sum = 0
        for i in range(int(licz)):
            sum = sum + X2[j + i]
            Sumlist2.append(sum)
        Xlist2.append(sum)
        j = j + int(licz)

    #niezmienione
    # Sumlist2 = []
    # for n in range(len(B)):
    #     sum = 0
    #     for j in range(z2, z1):
    #         sum = sum + X2[j]
    #         Sumlist2.append(sum)
    #     Xlist2.append(sum)


    Xlist1=np.array(Xlist1)
    Xlist1=-Xlist1
    Xlist2 = np.array(Xlist2)

    Xlist = []


    for n in range(len(B)):
        Xlist.append(Xlist1[n] + Xlist2[n])
    # print(Xlist1)
    # print(Xlist2)
    # print(Xlist)

    Clist = []
    for n in range(len(B)):
        if Xlist[n] > 0:
            Clist.append(1)
        else:
            Clist.append(0)
    #print(Clist)

    return Clist

def kodHamming1(B):
    Blist = [None] * 7
    Blist[2] = B[0]
    Blist[4] = B[1]
    Blist[5] = B[2]
    Blist[6] = B[3]

    Blist[0] = Blist[2] ^ Blist[4] ^ Blist[6]
    Blist[1] = Blist[2] ^ Blist[5] ^ Blist[6]
    Blist[3] = Blist[4] ^ Blist[5] ^ Blist[6]

    #    print(Blist)
    return Blist


def deKodHamming1(Blist):
    x1a = Blist[2] ^ Blist[4] ^ Blist[6]
    x1b = Blist[2] ^ Blist[5] ^ Blist[6]
    x1c = Blist[4] ^ Blist[5] ^ Blist[6]

    X1A = x1a ^ Blist[0]
    X1B = x1b ^ Blist[1]
    X1C = x1c ^ Blist[3]
    S = X1A * 2 ** 0 + X1B * 2 ** 1 + X1C * 2 ** 2
    # print(S)
    return S


def res(number):
    result = [int(i) for i in list('{0:0b}'.format(number))]
    for i in range(3):
        if len(result) < 4:
            result.insert(0, 0)

    return result


def KodHamming2(B):
    mList = []
    kList = []
    Glist = []

    I = np.eye(11)
    Plist = []

    for i in range(1, 16):
        Plist.append(res(i))
    Plist = np.array(Plist)

    Plist = np.delete(Plist, [0, 1, 3, 7], axis=0)
    Plist = np.fliplr(Plist)

    # print(Plist)

    G = np.hstack((Plist, I))
    # print(G)

    c = B @ G
    c[0] = c[0] % 2
    c[1] = c[1] % 2
    c[2] = c[2] % 2
    c[3] = c[3] % 2
    print(c)
    return c


def deKodHamming2(c):
    I = np.eye(4)
    Plist = []

    for i in range(1, 16):
        Plist.append(res(i))
    Plist = np.array(Plist)
    Plist = np.delete(Plist, [0, 1, 3, 7], axis=0)
    Plist = np.fliplr(Plist)

    # print(Plist)
    H = np.hstack((I, Plist.T))
    # print(H)

    s = np.dot(c, H.T)
    s[0]= s[0]%2
    s[1] = s[1] % 2
    s[2] = s[2] % 2
    s[3] = s[3] % 2

    s=s[0]*2**0+s[1]*2**1+s[2]*2**2+s[3]*2**3
    print(s)
    return s


def systemTransmisji(B,mod='ASK',hamming=1,zad=1,alfa=1,beta=0,konf=1):

    print(f'Bity:{B}')
    Blist=[]
    Bbeforemod = []
    if(hamming==1):
        j = 0
        for n in range(int(len(B)/4)):
            Bhelper = []
            for i in range(4):
                Bhelper.append(B[i+j])
            #print(Bhelper)
            Blist.append(kodHamming1(Bhelper))
            j=j+4
        #print(len(Blist))
        print(f'Blist={Blist}')
        #print(len(Blist))
        for n in range(len(Blist)):
            Bbeforemod=Bbeforemod+Blist[n]
        #print(len(Bbeforemod))
        print(f'Bbeforemod={Bbeforemod}')
    else:
        Blist=KodHamming2(B)

    if (zad==1):
        BafterMod=[]
        if mod=='ASK':
            BafterMod=demodASK(modASK(Bbeforemod),Bbeforemod)
            #print(demodASK(modASK(Bmod),Bmod))
            #print('ask')
        if mod=='PSK':
            print(Bbeforemod)
            BafterMod=demodPSK(modPSK(Bbeforemod),Bbeforemod)
            #print(demodPSK(modPSK(Bbeforemod),Bbeforemod))
            #print('psk')
        if mod=='FSK':
            BafterMod=demodFSK(modFSK(Bbeforemod),Bbeforemod)
            #print(demodFSK(modFSK(Bbeforemod),Bbeforemod))
            #print('fsk')

        print(f'Baftermod={BafterMod}')


        Bdekod = []
        zakres=int(len(BafterMod)/7)
        z1=0
        z2=-7
        for n in range(zakres):
            BdekodHelper = []
            z1=z1+7
            z2=z2+7
            for i in range(z2,z1):
                BdekodHelper.append(BafterMod[i])
            Bdekod.append(BdekodHelper)
        #print(Bdekod)
        finalB=[]
        for i in range(len(Bdekod)):
            if(deKodHamming1(Bdekod[i])==0):
                finalB.append(Bdekod[i][2])
                finalB.append(Bdekod[i][4])
                finalB.append(Bdekod[i][5])
                finalB.append(Bdekod[i][6])
            else:
                n=deKodHamming1(Bdekod[i])-1
                if Bdekod[i][n]==0:
                    Bdekod[i][n] = 1
                else:
                    Bdekod[i][n] = 0
                finalB.append(Bdekod[i][2])
                finalB.append(Bdekod[i][4])
                finalB.append(Bdekod[i][5])
                finalB.append(Bdekod[i][6])
        print(f'finalB={finalB}')

    if (zad == 2):
        Bmod = []
        BafterMod = []
        if mod == 'ASK':
            z=modASK(Bbeforemod)
            for i in range(len(z)):
                g=np.random.uniform(-1,1)
                g=alfa*g
                Bmod.append(z[i]+g)
            BafterMod = demodASK(Bmod, Bbeforemod)
        if mod == 'PSK':
            z = modPSK(Bbeforemod)
            for i in range(len(z)):
                g=np.random.uniform(-1,1)
                g=alfa*g
                Bmod.append(z[i]+g)
            BafterMod = demodPSK(Bmod,Bbeforemod)
        if mod == 'FSK':
            z = modFSK(Bbeforemod)
            for i in range(len(z)):
                g = np.random.uniform(-1, 1)
                g = alfa * g
                Bmod.append(z[i] + g)
            BafterMod = demodFSK(Bmod, Bbeforemod)
        print(f'Baftermod={BafterMod}')

        Bdekod = []
        zakres = int(len(BafterMod) / 7)
        z1 = 0
        z2 = -7
        for n in range(zakres):
            BdekodHelper = []
            z1 = z1 + 7
            z2 = z2 + 7
            for i in range(z2, z1):
                BdekodHelper.append(BafterMod[i])
            Bdekod.append(BdekodHelper)
        print(f'Bdekod={Bdekod}')
        #print(len(Bdekod))
        finalB = []
        for i in range(len(Bdekod)):
            if (deKodHamming1(Bdekod[i]) == 0):
                finalB.append(Bdekod[i][2])
                finalB.append(Bdekod[i][4])
                finalB.append(Bdekod[i][5])
                finalB.append(Bdekod[i][6])
            else:
                n = deKodHamming1(Bdekod[i])-1
                if Bdekod[i][n] == 0:
                    Bdekod[i][n] = 1
                else:
                    Bdekod[i][n] = 0
                finalB.append(Bdekod[i][2])
                finalB.append(Bdekod[i][4])
                finalB.append(Bdekod[i][5])
                finalB.append(Bdekod[i][6])
        print(f'finalB={finalB}')
        N=0
        for i in range(len(finalB)):
            if finalB[i]!=B[i]:
                N+=1

        print(f'Liczba zle przetransmitowanych bitow={N} BER={N/len(finalB)}')
        return finalB

    if (zad == 3):
        Bmod = []
        BafterMod = []
        if mod == 'ASK':
            z=modASK(Bbeforemod)
            t=[]
            for i in range(0, len(z)):
                t.append(i / fs)
            for i in range(len(z)):
                g=np.exp(-beta*t[i])
                Bmod.append(z[i]*g)
            BafterMod = demodASK(Bmod, Bbeforemod)
        if mod == 'PSK':
            z = modPSK(Bbeforemod)
            t = []
            for i in range(0, len(z)):
                t.append(i / fs)
            for i in range(len(z)):
                g = np.exp(-beta * t[i])
                Bmod.append(z[i] * g)
            BafterMod = demodPSK(Bmod,Bbeforemod)
        if mod == 'FSK':
            z = modFSK(Bbeforemod)
            t = []
            for i in range(0, len(z)):
                t.append(i / fs)
            for i in range(len(z)):
                g = np.exp(-beta * t[i])
                Bmod.append(z[i] * g)
            BafterMod = demodFSK(Bmod, Bbeforemod)
        print(f'Baftermod={BafterMod}')

        Bdekod = []
        zakres = int(len(BafterMod) / 7)
        z1 = 0
        z2 = -7
        for n in range(zakres):
            BdekodHelper = []
            z1 = z1 + 7
            z2 = z2 + 7
            for i in range(z2, z1):
                BdekodHelper.append(BafterMod[i])
            Bdekod.append(BdekodHelper)
        print(f'Bdekod={Bdekod}')
        #print(len(Bdekod))
        finalB = []
        for i in range(len(Bdekod)):
            if (deKodHamming1(Bdekod[i]) == 0):
                finalB.append(Bdekod[i][2])
                finalB.append(Bdekod[i][4])
                finalB.append(Bdekod[i][5])
                finalB.append(Bdekod[i][6])
            else:
                n = deKodHamming1(Bdekod[i])-1
                if Bdekod[i][n] == 0:
                    Bdekod[i][n] = 1
                else:
                    Bdekod[i][n] = 0
                finalB.append(Bdekod[i][2])
                finalB.append(Bdekod[i][4])
                finalB.append(Bdekod[i][5])
                finalB.append(Bdekod[i][6])
        print(f'finalB={finalB}')
        N=0
        for i in range(len(finalB)):
            if finalB[i]!=B[i]:
                N+=1

        print(f'Liczba zle przetransmitowanych bitow={N} BER={N/len(finalB)}')
        return finalB

    if (zad == 4):
        if konf==1:
            B4=systemTransmisji(systemTransmisji(B,mod=mod,zad=2,alfa=alfa,beta=beta),mod=mod,zad=3,alfa=alfa,beta=beta)
            N = 0
            for i in range(len(B)):
                if B4[i] != B[i]:
                    N += 1
            print(f'Liczba zle przetransmitowanych bitow={N} BER={N / len(B4)}')
        if konf == 2:
            B4 = systemTransmisji(systemTransmisji(B, mod=mod, zad=3, alfa=alfa, beta=beta), mod=mod, zad=2, alfa=alfa,
                                  beta=beta)
            N = 0
            for i in range(len(B)):
                if B4[i] != B[i]:
                    N += 1
            print(f'Liczba zle przetransmitowanych bitow={N} BER={N / len(B4)}')












B=str2bits('ABCDEFGH')
#B=[0,0,0,1,0,1,1,1,1,0,0,0,1,0,1,1,1,1,0,0,0,1,0,1,1,1,1]
# print(B)
# print(len(B))
#print(modASK(B))
#demodASK(modASK(B))

#demodASK(modASK(B))
# demodPSK()
# print(B)
# print(len(B))
# print(demodASK(modASK(B)))
# print(demodPSK(modPSK(B)))
# print(demodFSK(modFSK(B)))

#systemTransmisji(B,mod='FSK',zad=3,alfa=2,beta=20)
systemTransmisji(B,mod='ASK',zad=4,alfa=2,beta=20,konf=2)