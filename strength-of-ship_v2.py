"""
Created on Fri Nov 19 20:11:05 2021
@author: nedir ymamov
"""


import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt


def bonjeanAlani(offset, posta0, posta, suhatti):
    offset_new = np.zeros((101, 8))
    for i in range(8):
        f = interp1d(posta0, offset[:, i], kind = "cubic")
        offset_new[:, i] = f(posta)
    alan = np.zeros((101, 8))   # BON-JEAN ALANLARI
    for i in range(101):
        alan[i, 1:] = 2 * cumtrapz(offset_new[i, :], suhatti[:])
    return alan


def prohaskaDagilimi(boy, deplasman):
    qx = np.zeros(101)   # TOPLAM GEMİ AĞIRLIK DAĞILIMI (PROHASKA YÖNTEMİ)
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
    return qx


def ataletDagilimi(boy, posta):
    Ix = np.zeros(101)   # ATALET MOMENT DAĞILIMI
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
    return Ix


def dalgaKaydirma(boy, draft, deplasman, H, alan, posta, suhatti, dalga_katsayi, yogunluk):
    posta1 = np.array([0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]) * boy / 10
    c = np.interp(posta[:51], posta1, dalga_katsayi)   # YARI DALGA DEĞERLERİ
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
    return ax


def gerilmeHesabi(name, boy, w, ax, qx, Ix, posta, ymax):
    px = ax - qx
    dpx0 = np.array([0, *cumtrapz(px, posta)])
    dpx = dpx0 - dpx0[-1] * posta / boy  # LİNEER DÜZELTME (%0.03dpx.son<dpx.max [M. Savcı])
    ddpx = np.array([0, *cumtrapz(dpx, posta)])
    if name == "cukur":   # ARTIK MOMENT DÜZELTME [M. Savcı 1980]
        C = 1 - np.cos(2 * np.pi * posta / boy)
        d1Qx = -(ddpx[-1] * C) / boy
        Qx = dpx - d1Qx
        Mx = ddpx + np.array([0, *cumtrapz(d1Qx, posta)])
    else:
        dax = np.array([0, *cumtrapz(ax, posta)])
        Qx = dpx + (-ddpx / w) * ax
        Mx = ddpx + (-ddpx[-1] / w) * dax
    W = Ix / ymax   # SECTİON MODULUS
    gerilme = np.array([0, *(9.81 * Mx[1 : -1]) / (W[1 : -1] * 1000), 0])   # 1[ton] = 9.81[kN]
    return Qx, Mx, gerilme


def grafikCizimi(name, ax, qx, Qx, Mx, gerilme, posta):
    plt.figure(figsize = (10, 4), facecolor = "seagreen")
    plt.title(name, fontweight = "bold") # GRAFİĞİN BAŞLIĞI
    plt.xlabel("Gemi Boyu [m]", fontweight = "bold")
    plt.plot(posta, ax, posta, qx, posta, Qx, posta, Mx, posta, gerilme)
    plt.legend(["a [ton/m]", "q [ton/m]", "Q [ton]", "M [ton.m]", "σ [MPa]"], loc = "best")
    plt.plot([-1, boy + 1], [0, 0])
    plt.grid(color = "green", linestyle = "--", linewidth = .5)
    plt.show()

def hesapla(offset, boy, genislik, draft, yogunluk):
    H = boy / 20 # DALGA YÜKSEKLİĞİ
    blok_katsayi = .7
    deplasman = boy * genislik * draft * blok_katsayi * yogunluk
    posta0 = np.array([0, .5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10]) * boy / 10
    posta = np.linspace(0, boy, 101)
    suhatti = np.array([0, .3, 1, 2, 3, 4, 5, 6]) * draft / 4
    ymax = 1.5 * draft - .4 * 1.5 * draft    # TARAFSIZ EKSEN KABULU [İ. Bayer]
    
    alan = bonjeanAlani(offset, posta0, posta, suhatti)
    qx = prohaskaDagilimi(boy, deplasman) # GEMİNİN AĞITLIK DAĞILIMI
    Ix = ataletDagilimi(boy, posta) # GEMİNİN ATALET DAĞILIMI
    
    # SAKİN SU
    ax = 1.005 * alan[:, 5] * yogunluk
    Qx, Mx, gerilme = gerilmeHesabi("sakin", boy, deplasman, ax, qx, Ix, posta, ymax)
    # BURADA Qx, Mx VE gerilme 'Yİ BELİRLİ BİR ORAN İLE BÖLÜNÜR
    # BU ORANLARI KENDİ DEĞERLERİNİZ İLE DEĞİŞTİRMEYİ UNUTMAYINIZ
    grafikCizimi("SAKİN SU", ax, qx, Qx / 5, Mx / 60, gerilme / 6, posta)
    
    # DALGA ÇUKURU
    dalga_katsayi = [1, .966, .871, .795, .578, .422, .28, .16, .072, .018, 0]
    ax = dalgaKaydirma(boy, draft, deplasman, H, alan, posta, suhatti, dalga_katsayi, yogunluk)
    Qx, Mx, gerilme = gerilmeHesabi("cukur", boy, deplasman, ax, qx, Ix, posta, ymax)
    grafikCizimi("DALGA ÇUKURU", ax, qx, Qx / 5, Mx / 20, gerilme / 2, posta)
    
    # DALGA TEPESİ
    dalga_katsayi = np.flipud(dalga_katsayi) # DALGA KATSAYISININ LİSTESİNİ TERSİNE ÇEVİRME
    ax = dalgaKaydirma(boy, draft, deplasman, H, alan, posta, suhatti, dalga_katsayi, yogunluk)
    Qx, Mx, gerilme = gerilmeHesabi("tepe", boy, deplasman, ax, qx, Ix, posta, ymax)
    grafikCizimi("DALGA TEPESİ", ax, qx, Qx / 7, Mx / 110, gerilme / 10, posta)


boy = 87
genislik = 10.88
draft = 4.35
yogunluk = 1.025
offset = np.loadtxt("s60.txt", dtype = float) * genislik / 2

hesapla(offset, boy, genislik, draft, yogunluk)