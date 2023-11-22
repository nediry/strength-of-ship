"""
Created on Fri Nov 19 22:23:32 2021
@author: nedir ymamov
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

boy = 69
genislik = boy / 8
draft = genislik / 2.5
derinlik = 1.5 * draft
H = boy / 20 # DALGA YÜKSEKLİĞİ
CB = .7
yogunluk = 1.025
posta = np.linspace(0, boy, 101)
deplasman = boy * genislik * draft * CB * yogunluk
offset = np.loadtxt("s60.txt", dtype = float)
ymax = derinlik - .4 * derinlik # TARAFSIZ EKSEN KABULU [İ. Bayer]

suhatti = np.array([0, .3, 1, 2, 3, 4, 5, 6]) * draft / 4

offset_new = np.zeros((101, 8)) # GEMİYİ 100 POSTAYA BÖLÜNÜR
posta0 = np.array([0, .5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10]) * boy / 10
for i in range(8):
    f = interp1d(posta0, offset[:, i], kind = "cubic")
    offset_new[:, i] = f(posta)
offset_new *= genislik / 2
alan = np.zeros((101, 8)) # BON-JEAN ALANLARI
for i in range(101):
    alan[i, 1:] = 2 * cumtrapz(offset_new[i, :], suhatti[:])

qx = np.zeros(101) # TOPLAM GEMİ AĞIRLIK DAĞILIMI (PROHASKA YÖNTEMİ)
a = .68 * deplasman / boy
b = 1.185 * deplasman / boy
c = .58 * deplasman / boy
n = 1
for i in range(101):
    if i < 34:
        qx[i] = a + i * (b - a) / 34
    if i > 33 and i < 68:
        qx[i] = b
    if i > 67:
        qx[i] = b - n * (b - c) / 33
        n += 1

Ix = np.zeros(101) # ATALET MOMENT DAĞILIMI
Wmax = .193
Iy = 3 * Wmax * boy / 100  # ORTA KESİT ATALET MOMENTİ [m4]
for i in range(100):
    if posta[i] <= boy / 20:
        Ix[i] = 5 * Iy * posta[i] / boy
    if boy / 20 < posta[i] <= 7 * boy / 20:
        Ix[i] = .25 * Iy + (15 * Iy) * (posta[i] - boy / 20) / (6 * boy)
    if 7 * boy / 20 < posta[i] <= 15 * boy / 20:
        Ix[i] = Iy
    if 15 * boy / 20 < posta[i] <= 19 * boy / 20:
        Ix[i] = Iy - 2.5 * (Iy / boy) * (posta[i] - 15 * boy / 20)
    if posta[i] > 19 * boy / 20:
        Ix[i] = .5 * Iy - 10 * (Iy / boy) * (posta[i] - 19 * boy / 20)

ax = alan[:, 5] * yogunluk
px = ax - qx
dpx0 = np.array([0, *cumtrapz(px, posta)])
dpx = dpx0 - dpx0[-1] * posta / boy  # LİNEER DÜZELTME (%0.03dpx.son<dpx.max [M. Savcı])
ddpx = np.array([0, *cumtrapz(dpx, posta)])
dax = np.array([0, *cumtrapz(ax, posta)]) # ARTIK MOMENT DÜZELTME [M. Savcı 1980]
Qx = dpx + (-ddpx[-1] / deplasman) * ax
Mx = ddpx + (-ddpx[-1] / deplasman) * dax
W = Ix / ymax # KESİT MODÜLÜ
gerilme = np.array([0, *(9.81 * Mx[1 : -1]) / (W[1 : -1] * 1000), 0]) # 1[ton] = 9.81[kN]

plt.figure(figsize = (10, 4))
plt.title("SAKİN SU", fontweight = "bold")
plt.xlabel("Gemi Boyu [m]", fontweight = "bold")
plt.plot(posta, ax, posta, qx, posta, Qx / 3, posta, Mx / 50, posta, gerilme / 3)
plt.legend(["a[ton/m]", "q[ton/m](", "Q[ton]", "M[ton.m]", "σ[MPa]"], loc = "best")
plt.grid(color = "green", linestyle = "--", linewidth = .7)
plt.plot([-1, 70], [0, 0])
plt.show()

# DALGA ÇUKURU

posta1 = np.array([0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]) * boy / 10
dalga_katsayi = [1, .966, .871, .795, .578, .422, .28, .16, .072, .018, 0]
c = np.interp(posta[:51], posta1, dalga_katsayi)   # YARIM DALGA DEĞERLERİ
c = np.concatenate((np.delete(c, -1), np.flipud(c)), axis = 0) # YARI DALGAYI TAM YAPMA
ksi = draft - H / 2 + H * c

ax = np.zeros(101)   # DALGAYI DİKEY KAYDIRARAK a(x) BULMAK
for i in range(200):
    for j in range (101):
        ax[j] = yogunluk * np.interp(ksi[j], suhatti, alan[j, :])
    if round(deplasman) < round(np.trapz(ax, posta)):
        ksi -= .004
    elif round(deplasman) > round(np.trapz(ax, posta)):
        ksi += .004
    else: break

px = ax - qx
dpx0 = np.array([0, *cumtrapz(px, posta)])
dpx = dpx0 - dpx0[-1] * posta / boy  # LİNEER DÜZELTME (%0.03dpx.son<dpx.max [M. Savcı])
ddpx = np.array([0, *cumtrapz(dpx, posta)]) 
C = 1 - np.cos(2 * np.pi * posta / boy)   # ARTIK MOMENT DÜZELTME [M. Savcı 1980]
d1Qx = - (ddpx[-1] * C) / boy
Qx = dpx - d1Qx
Mx = ddpx + np.array([0, *cumtrapz(d1Qx, posta)])
W = Ix / ymax # KESİT MODÜLÜ
gerilme = np.array([0, *(9.81 * Mx[1 : -1]) / (W[1 : -1] * 1000), 0])   # 1[ton] = 9.806[kN]

plt.figure(figsize = (10, 4))
plt.title("DALGA ÇUKURU", fontweight = "bold")
plt.xlabel("Gemi Boyu [m]", fontweight = "bold")
plt.plot(posta, ax, posta, qx, posta, Qx / 4, posta, Mx / 20, posta, gerilme / 2)
plt.legend(["a[ton/m]", "q[ton/m]", "Q[ton]", "M[ton.m]", "σ[MPa]"], loc = "best")
plt.grid(color = "green", linestyle = "--", linewidth = .7)
plt.plot([-1, 70], [0, 0])
plt.show()

# DALGA TEPESİ

dalga_katsayi = [0, .018, .072, .16, .28, .422, .578, .795, .871, .966, 1]
c = np.interp(posta[:51], posta1, dalga_katsayi)   #YARIM DALGA DEĞERLERİ
c = np.concatenate((np.delete(c, -1), np.flipud(c)), axis = 0) #YARI DALGAYI TAM YAPMA
ksi = draft - H / 2 + H * c

ax = np.zeros(101)   # DALGAYI DİKEY KAYDIRARAK a(x) BULMAK
for i in range(200):
    for j in range (101):
        ax[j] = yogunluk * np.interp(ksi[j], suhatti, alan[j, :])
    if round(deplasman) < round(np.trapz(ax, posta)):
        ksi -= .004
    elif round(deplasman) > round(np.trapz(ax, posta)):
        ksi += .004
    else: break

px = ax - qx
dpx0 = np.array([0, *cumtrapz(px, posta)])
dpx = dpx0 - dpx0[-1] * posta / boy  # LİNEER DÜZELTME (%0.03dpx.son<dpx.max [M. Savcı])
ddpx = np.array([0, *cumtrapz(dpx, posta)])
dax = np.array([0, *cumtrapz(ax, posta)]) # ARTIK MOMENT DÜZELTME [M. Savcı 1980]
Qx = dpx + (-ddpx / deplasman) * ax
Mx = ddpx + (-ddpx[-1] / deplasman) * dax
W = Ix / ymax # KESİT MODÜLÜ
gerilme = np.array([0, *(9.81 * Mx[1 : -1]) / (W[1 : -1] * 1000), 0])   # 1[ton] = 9.806[kN]

plt.figure(figsize = (10, 4))
plt.grid(color = "green", linestyle = "--", linewidth = .7)
plt.title("DALGA TEPESİ", fontweight = "bold")
plt.xlabel("Gemi Boyu [m]", fontweight = "bold")
plt.plot(posta, ax, posta, qx, posta, Qx / 6, posta, Mx / 100, posta, gerilme / 6)
plt.legend(["a[ton/m]", "q[ton/m]", "Q[ton]", "M[ton.m]", "σ[MPa]"], loc = "best")
plt.plot([-1, 70], [0, 0])
plt.show()
