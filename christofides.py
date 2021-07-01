#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:14:09 2021

@author: romain
"""
import itertools
import pandas as pd
import random
from collections import deque
import pytest
import pickle
import os

PICKLE_FILE_1 = 'data.dat'
PICKLE_FILE_2 = 'chemin.dat'
SIZE = 6921

distances_gpu = pd.read_pickle(PICKLE_FILE_1)  # .drop(['Unnamed: 0'], axis=1).rename(columns={'Unnamed: 0.1':'lieu'})
# distances_gpu = pd.read_excel('distances_gpu.xlsx').drop(['Unnamed: 0'], axis=1).rename(columns={'Unnamed: 0.1':'lieu'})
distances_gpu['lieu'] = distances_gpu.columns
distances_gpu.index = distances_gpu['lieu']
distances_gpu.drop(['lieu'], axis=1, inplace=True)
global d
d = distances_gpu.iloc[:SIZE, :SIZE].copy()


def add_to_pickle(path, item):
    with open(path, 'ab') as file:
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
    print('soit : ', int(km), 'km')
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
    return impaire_d


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
        # print(pos_chem)
        next_try = impaire_d[[next_ville]].sort_values(ascending=True, by=[next_ville], axis=0, kind='mergesort')
        for proxima in range(1, len(impaire_d)):  # Pour la taille
            # print(proxima, len(next_try), next_try.index)
            # print(next_try, proxima)
            next_ville = next_try.iloc[proxima:].index[0]
            if len(imp_visite) == len(impaire_d):
                print('Chemin impair de poids minimum terminé')
                break
            elif imp_set_visite.isdisjoint(set({next_ville})):  # not in visite.values()
                imp_visite.update({pos_chem + 2: next_ville})
                imp_set_visite.add(next_ville)
                #  print(pos_chem, imp_visite[pos_chem])
                imp_km += impaire_d.loc[[imp_visite[pos_chem]], [imp_visite[pos_chem + 2]]].values[0][0]
                break
    print('soit : ', int(imp_km), 'km')
    return imp_visite, imp_km


def liaison(visite, imp_visite):
    union = deque()
    for k, v in visite.items():
        if k % 2 == 0:
            union.append(v)
        if k % 2 == 1:
            if v == imp_visite[k]:
                union.append(v)
            if v != imp_visite[k]:
                double_edge = list()
                double_edge.append(v)
                double_edge.append(imp_visite[k])
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
        self.road = list()
        self.road.append(self.start)
        self.pos_set = [x for x in range(len(union)) if isinstance(union[x], list)]
        self.vertex_deque: deque = deque(self.start)
        self.union = pd.DataFrame(list(union), columns=['lieu'])
        self.km_final = float(0)

    def add_Edge(self, it):
        # print(edge)
        # print(self.pos_set)
        if it != 0:
            if it not in self.pos_set:
                self.road.append(self.union.iloc[it].values[0])
                # print('it', it, 'union', self.union.iloc[it])
                self.km_final += self.dist_loc(self.road[it - 1], self.union.iloc[it].values[0])
            # print(lieu_str)
            elif it in self.pos_set:
                # print('LEGO')
                self.euler_set = set()
                self.lieu_set = self.ret_lieu_str(it)  # self.union.iloc[it].loc['lieu']
                self.lieu_0 = self.ret_lieu_str(it - 1)
                self.lieu_1 = self.lieu_set[0]
                self.lieu_2 = self.lieu_set[1]
                self.lieu_3 = self.ret_lieu_str(it + 1)
                self.euler_tour()
                self.best_dist()
                for k in range(len(self.best_euler_list)):
                    self.road.append(self.best_euler_list[k])
                self.km_final += self.km_best
                # print(euler_tour)

    # else:
    #  self.road.append(self.union.iloc[it].values[0])
    # print('lieu: ',self.union.iloc[it].values[0])
    def best_dist(self):
        height_euler_perm = len(self.euler_perm)
        self.best_euler_list = list()
        self.km_best = float(999)
        for i in range(height_euler_perm):
            self.best_euler_set = self.euler_perm.pop()
            self.euler_list_cur = list()
            self.euler_list_cur.append(self.lieu_0)
            self.euler_list_cur.append(self.best_euler_set[0])
            #   print('voila', self.euler_list_cur)
            self.km_cur = float(0)
            for j in range(len(self.best_euler_set) - 1):
                # print(len(self.best_euler_set))
                #   print(self.best_euler_set[1])
                #  print(type(self.best_euler_set))
                self.euler_list_cur.append(self.best_euler_set[j + 1])
                self.km_cur += self.dist_loc(self.lieu_0, self.best_euler_set[j])
                self.km_cur += self.dist_loc(self.best_euler_set[j], self.best_euler_set[j + 1])
            # print(self.km_cur)
            # print('km_cur: ',self.km_cur, 'KM : ', self.km_best, 'euler: ', self.best_euler_list, 'euler_cur: ', self.euler_list_cur  )
            if self.km_cur < self.km_best:
                self.km_best = self.km_cur
                self.best_euler_list = self.euler_list_cur

            # for k in range(len(self.best_dist)):
        # print('best : ', self.best_euler_list)

    def ret_lieu_str(self, it):
        return self.union.iloc[it].loc['lieu']

    def twice_of_set(self, edge_set):
        if isinstance(edge_set, set):
            self.voie_1 = edge_set[0][0]
            self.voie_2 = edge_set[0][1]

    def dist_loc(self, a, b):
        # print('values' , d.loc[[a],[b]].values[0][0])
        return d.loc[[a], [b]].values[0][0]

    def euler_tour(self):
        self.euler_set.add(self.lieu_1)
        self.euler_set.add(self.lieu_2)
        self.euler_set.add(self.lieu_3)
        self.euler_perm = set()
        self.euler_perm = {p for p in itertools.permutations(self.euler_set)}

    def __str__(self):
        return 'Nombre de kilomètre : ' + str(self.km_final) \
               + '\nChemin de départ : ' + str(self.road) \
               + '\nPas deux fois le même lieu : ' + str(
            len(pd.DataFrame(self.road, columns=['lieu'])['lieu'].unique()) == SIZE)


def nothing():
    list_pm, km = poids_min(d)
    # print(list_pm)
    list_imp = impair(list_pm)
    # print('imp : ' , list_imp)
    list_imp_pm, km_imp = poids_min_impair(list_imp)
    union = liaison(list_pm, list_imp_pm)
    # print(union)
    # liaison(poids_min(d),impair(poids_min(d))
    chemin = Graph(union)
    for i in range(SIZE):
        chemin.add_Edge(i)
    print('fin')
    print(chemin)
    add_to_pickle(PICKLE_FILE_2, chemin.road)


nothing()
