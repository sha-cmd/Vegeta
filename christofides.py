#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:14:09 2021

@author: romain
"""
import itertools
import pandas as pd
import numpy as np
import random
from collections import deque
import pytest
import pickle
import os

PICKLE_FILE_1 = 'data.dat'
PICKLE_FILE_2 = 'chemin.dat'
SIZE = 6921

global d  # Notre table de données

distances_gpu = pd.read_pickle(PICKLE_FILE_1)  # .drop(['Unnamed: 0'], axis=1).rename(columns={'Unnamed: 0.1':'lieu'})
distances_gpu['lieu'] = distances_gpu.columns
distances_gpu.index = distances_gpu['lieu']
distances_gpu.drop(['lieu'], axis=1, inplace=True)
d = distances_gpu.iloc[:SIZE, :SIZE].copy()


def write_to_pickle(path, item):
    with open(path, 'wb') as file:
        pickle.dump(item, file, pickle.HIGHEST_PROTOCOL)


def poids_min(d):
    visite = dict()
    set_visite = set()
    start = d.columns[0]  # random.choice(d.columns)
    visite.update({0: start})
    set_visite.add(start)
    next_ville = start
    next_try = ''
    km = float()
    for pos_chem in range(SIZE):
        next_try = d[[next_ville]].sort_values(ascending=True, by=[next_ville], axis=0, kind='mergesort')
        for proxima in range(SIZE):  # Pour la taille
            next_ville = next_try.iloc[proxima:].index[0]
            if SIZE == len(visite):
                print('Chemin de poids minimum terminé')
                break
            elif set_visite.isdisjoint(set({next_ville})):  # not in visite.values()
                visite.update({pos_chem + 1: next_ville})
                set_visite.add(next_ville)
                km += d.loc[[visite[pos_chem]], [visite[pos_chem + 1]]].values[0][0]
                break
    # Termine la boucle
    if pos_chem == SIZE - 1:
        km += d.loc[[visite[pos_chem]], [visite[0]]].values[0][0]
    return (visite, km)


def impair(visite):
    impaire_ensemble = dict()
    for v in range(len(visite)):
        if v % 2 == 1:
            impaire_ensemble.update({v: visite[v]})
    impaire_d = d[list(impaire_ensemble.values())].loc[list(impaire_ensemble.values())]
    imp_visite = dict()
    for i in range(1, len(impaire_d.index), 2):
        imp_visite.update({i: impaire_d.index[i]})
    return impaire_d, imp_visite


def poids_min_impair(impaire_d):
    # Nous prenons la ligne de la ville la plus proche
    imp_visite = dict()
    imp_set_visite = set()
    start = impaire_d.index[0]
    imp_visite.update({1: start})
    imp_set_visite.add(start)
    next_ville = start
    next_try = ''
    imp_km = float()
    for pos_chem in range(1, SIZE, 2):
        next_try = impaire_d[[next_ville]].sort_values(ascending=True, by=[next_ville], axis=0, kind='mergesort')
        for proxima in range(1, len(impaire_d)):  # Pour la taille
            next_ville = next_try.iloc[proxima:].index[0]
            if len(imp_visite) == len(impaire_d):
                print('Chemin impair de poids minimum terminé')
                break
            elif imp_set_visite.isdisjoint(set({next_ville})):  # not in visite.values()
                imp_visite.update({pos_chem + 2: next_ville})
                imp_set_visite.add(next_ville)
                imp_km += impaire_d.loc[[imp_visite[pos_chem]], [imp_visite[pos_chem + 2]]].values[0][0]
                break
    return imp_visite, imp_km


def liaison(visite, imp_visite):
    union = deque()
    for k, v in visite.items():
        if k % 2 == 0:
            union.append(v)
        if k % 2 == 1:
            if k == (len(visite.items())) - 2:
                union.append(visite[k])

            elif k < (len(visite.items())) - 2:
                double_edge = list()
                double_edge.append(visite[k])
                double_edge.append(visite[k+2])
                union.append(double_edge)
    return union


def dist(a, i, b, j):
    return d.loc[[a[i]], [b[j]]].values[0][0]


class Vertex:
    def __init__(self, name):
        self.name = name


class Edge:
    def __init__(self, one, two, dist):
        self.one = one
        self.two = two
        self.dist = dist


class Graph:
    def __init__(self, union):
        self.pos = int(0)
        self.start = union[0]
        self.union_deque = union
        self.road = list()
        self.pos_set = [x for x in range(len(union)) if isinstance(union[x], list)]
        self.vertex_deque: deque = deque(self.start)
        self.union_df = pd.DataFrame(list(union), columns=['lieu'])
        self.km_final = np.float64(0)
        self.best_euler_perm_list = list()
        self.lieux_list = list()
        self.impair_list = list()
        self.lieu_0 = str()
        self.lieu_1 = str()
        self.lieu_2 = str()
        self.lieu_3 = str()

    def add_edge(self, it):
        if self.union_deque[it] not in self.road:
            if it >= 1:
                if it == len(self.union_deque)-1:
                    self.lieu_0 = self.road[-1]
                    self.lieu_1 = self.ret_lieu_str(it)
                    if self.lieu_1 not in self.road:
                        self.km_final += self.dist_loc(self.lieu_0, self.lieu_1)
                        self.road.append(self.lieu_1)
                elif it in self.pos_set:
                    self.impair_list = self.ret_lieu_str(it)
                    self.lieu_0 = self.ret_lieu_str(it - 1)
                    self.lieu_1 = self.impair_list[0] if (self.impair_list[0] not in self.road) else ''
                    self.lieu_2 = self.impair_list[1] if (self.impair_list[1] not in self.road) else ''
                    self.lieu_3 = self.ret_lieu_str(it + 1) if (self.ret_lieu_str(it + 1) not in self.road) else ''
                    del self.lieux_list[:]
                    self.lieux_list.append(self.lieu_1)
                    self.lieux_list.append(self.lieu_2)
                    self.lieux_list = [x for x in self.lieux_list if x != '']
                    self.euler_tour()

                    self.best_dist()
                    if self.best_km > 0:
                        self.km_final += self.best_km
                    if it >= 1:
                        if self.lieu_0 not in self.road:
                            self.road.append(self.lieu_0)
                        for k in range(len(self.best_euler_perm_list)):
                            if self.best_euler_perm_list[k] not in self.road:
                                self.road.append(self.best_euler_perm_list[k])
                        if self.lieu_3 not in self.road:
                            self.road.append(self.lieu_3)

    def best_dist(self):
        height_euler_perm = len(self.euler_perms)
        del self.best_euler_perm_list[:]
        self.best_km = np.float64(-1)
        for i in range(height_euler_perm):
            euler_list_cur = list()
            euler_list_cur = self.euler_perms.pop()
            km_cur = np.float64(0)
            km_cur += self.dist_loc(self.lieu_0, euler_list_cur[0])
            for j in range(len(euler_list_cur) - 1):
                km_cur += self.dist_loc(euler_list_cur[j], euler_list_cur[j + 1])
                km_cur += self.dist_loc(euler_list_cur[j + 1], self.lieu_3)
                if self.best_km < 0:
                    self.best_km = km_cur
                    del self.best_euler_perm_list[:]
                    for k in range(len(euler_list_cur)):
                        self.best_euler_perm_list.append(euler_list_cur[k])

                elif km_cur < self.best_km:
                    del self.best_euler_perm_list[:]
                    self.best_km = km_cur
                    for k in range(len(euler_list_cur)):
                        self.best_euler_perm_list.append(euler_list_cur[k])
    def ret_lieu_str(self, it):
        return self.union_deque[it]

    def dist_loc(self, a, b):
        return d.loc[[a], [b]].values[0][0]

    def euler_tour(self):
        self.euler_set = set()
        for i in range(len(self.lieux_list)):
            self.euler_set.add(self.lieux_list[i])
        self.euler_perms = set()
        self.euler_perms = {p for p in itertools.permutations(self.euler_set)}

    def __str__(self):
        return 'Nombre de kilomètre : ' + str(int(self.km_final)) \
                + '\nChemin de départ : ' + str(self.road) \
                + '\n\nNombre de kilomètre par christofides : ' + str(int(self.km_final)) + 'km'\
                + '\nPas deux fois le même lieu : ' + str(
            len(pd.DataFrame(self.road, columns=['lieu'])['lieu'].unique()) == len(self.road)) + ', nombre de lieux totaux : ' + str(len(self.road))


def christofides():
    list_pm, km = poids_min(d)
    print('le chemin de poids nominal était de : ', km)
    list_imp, list_imp_sec = impair(list_pm)
    list_imp_pm, km_imp = poids_min_impair(list_imp)
    union = liaison(list_pm, list_imp_sec)
    chemin = Graph(union)
    for i in range(len(union)):
        chemin.add_edge(i)
    print('fin')
    print(chemin)
    print('le chemin de poids nominal était de : ', int(km), 'km')
    write_to_pickle(PICKLE_FILE_2, chemin.road)


christofides()
