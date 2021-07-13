#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:14:09 2021

@author: romain
"""
import itertools
import pandas as pd
import pandasql as ps
import numpy as np
import random
from collections import deque
import pytest
import pickle
import os
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

PICKLE_FILE_1 = 'data.dat'
PICKLE_FILE_2 = 'chemin.dat'
arr_list = [
        'PARIS 1ER ARRDT',
        'PARIS 2E ARRDT',
        'PARIS 3E ARRDT',
        'PARIS 4E ARRDT',
        'PARIS 5E ARRDT',
        'PARIS 6E ARRDT',
        'PARIS 7E ARRDT',
        'PARIS 8E ARRDT',
        'PARIS 9E ARRDT',
        'PARIS 10E ARRDT',
        'PARIS 11E ARRDT',
        'PARIS 12E ARRDT',
        'PARIS 13E ARRDT',
        'PARIS 14E ARRDT',
        'PARIS 15E ARRDT',
        'PARIS 16E ARRDT',
        'PARIS 17E ARRDT',
        'PARIS 18E ARRDT',
        'PARIS 19E ARRDT',
        'PARIS 20E ARRDT',
        'BOIS DE BOULOGNE',
        'BOIS DE VINCENNES',
        'HAUTS-DE-SEINE',
        'SEINE-SAINT-DENIS',
        'VAL-DE-MARNE']





def matrix():
    kernel_code_template = """
    __global__ void MatrixMulKernel(double* a, double* b, double* c)
    {

        for (int k = 0; k <= %(MATRIX_SIZE)s; ++k) {
                for (int l = 0; l <= %(MATRIX_SIZE)s; ++l) {
                        double Aelement = a[l] - a[k];
                        double Belement = b[l] - b[k];
                        c[%(MATRIX_SIZE)s * k + l] = sqrt((Aelement*111)*(Aelement*111) + (Belement*80)*(Belement*80));
                        }
        }


    }
    """

    matrice = pd.read_excel('data/export_df_graph.xlsx')
    NOMBRE = len(matrice)
    MATRIX_SIZE = NOMBRE  # len(na.iloc[:NOMBRE])

    a_cpu = np.array(matrice['lat'].iloc[:NOMBRE]).reshape((NOMBRE, 1)).astype(np.float64())
    b_cpu = np.array(matrice['lon'].iloc[:NOMBRE]).reshape((NOMBRE, 1)).astype(np.float64())

    c_cpu = np.dot(a_cpu, b_cpu.T)
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)

    c_gpu = gpuarray.empty((NOMBRE, NOMBRE), np.float64())

    kernel_code = kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE
    }

    mod = compiler.SourceModule(kernel_code)

    matrixmul = mod.get_function("MatrixMulKernel")

    matrixmul(
        a_gpu, b_gpu,
        c_gpu,
        block=(5, 5, 1),
    )

    result = c_gpu.get()

    col_new = dict()
    for it, x in enumerate(list((matrice['lieu'].unique()))):
        col_new.update({it: x})

    df_result = pd.DataFrame(result)
    #print(df_result)
    df_result = df_result.rename(columns=col_new, index=col_new)
    write_to_pickle(PICKLE_FILE_1, df_result)


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
                # Chemin de poids minimum calculé
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
                # Chemin impair de poids minimum terminé
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
        return 'Nombre de kilomètre par Christofides : ' + str(int(self.km_final)) + 'km'\
                + '\nChemin de départ : ' + str(self.road) \
                + '\n\nNombre de kilomètre par Christofides : ' + str(int(self.km_final)) + 'km'\
                + '\nPas deux fois le même lieu : ' + str(
            len(pd.DataFrame(self.road, columns=['lieu'])['lieu'].unique()) == len(self.road)) + ', nombre de lieux totaux : ' + str(len(self.road))


def christofides(q_h=0.95, q_b=0.05, col='soin'):#, motif='à surveiller'):
    #data_orig = pd.read_csv('data.csv', sep=';')
    data_surv_lieu = pd.read_excel('data/quant_surveiller_lieu_q_h_'+ str(q_h) + '_' + str(q_b) + '.xlsx')
    data_orig = pd.read_excel('data/new_data_end_' + str(q_h) + '_' + str(q_b) + '.xlsx').drop(['Unnamed: 0', 'id'],axis=1)
    print(data_orig.columns)
    #data = data_orig.loc[data_orig['genre'] == genre].copy()
    data = pd.DataFrame()
    if col == 'soin':
        data = data_orig.loc[data_orig[col] == 'à surveiller'].copy()
    elif col == 'ratio':
        print('OK')
        data = data_orig.loc[data_orig[col] < 0.5].copy()
        print(data['lieu'].count())
    elif col == 'remarquable':
        data = data_orig.loc[data_orig[col] == 1].copy()
    data = data.drop_duplicates(subset='lieu', ignore_index=True)
    print(data['lieu'].count())
#    data.drop('id', axis=1, inplace=True)
 #   data.drop('numero', axis=1, inplace=True)
 #   mask = data['variete'] == ''
 #   data.loc[mask, 'variete'] = 'n. sp.'
 #   mask = data['espece'] == ''
 #   data.loc[mask, 'espece'] = 'n. sp.'
 #   mask = data['domanialite'] == ''
 #   data.loc[mask, 'domanialite'] = 'Jardin'
 #   data['remarquable'].fillna(value=0, inplace=True)
 #   data['remarquable'] = data['remarquable'].map({0.: False, 1.: True})
 #   data['remarquable'] = data['remarquable'].convert_dtypes()
 #   colonnes = ['arrondissement', 'type_emplacement', 'lieu', 'id_emplacement']
 #   for col in colonnes:
 #       data[col] = data[col].convert_dtypes(convert_string=True)
 #   colonnes = ['stade_developpement', 'espece', 'variete', 'genre', 'libelle_francais', \
  #              'complement_addresse', 'domanialite']
 #   for col in colonnes:
  #      data[col].fillna(value="", inplace=True)
   #     data[col] = data[col].convert_dtypes(convert_string=True)
    lieu_aire = pd.DataFrame(data.pivot_table(index=['lieu'], aggfunc='size'), columns=['aire'])
    x = lieu_aire.reset_index()
    q7 = """SELECT  data.lieu as lieu,
                    data.arrondissement as arrond,
                    data.geo_point_2d_a as lat,
                    data.geo_point_2d_b as lon,
                    x.aire as aire
                    FROM data INNER JOIN x ON data.lieu == x.lieu ORDER BY lieu"""
    df_graph = ps.sqldf(q7, locals())
    arr_list = [
        'PARIS 1ER ARRDT',
        'PARIS 2E ARRDT',
        'PARIS 3E ARRDT',
        'PARIS 4E ARRDT',
        'PARIS 5E ARRDT',
        'PARIS 6E ARRDT',
        'PARIS 7E ARRDT',
        'PARIS 8E ARRDT',
        'PARIS 9E ARRDT',
        'PARIS 10E ARRDT',
        'PARIS 11E ARRDT',
        'PARIS 12E ARRDT',
        'PARIS 13E ARRDT',
        'PARIS 14E ARRDT',
        'PARIS 15E ARRDT',
        'PARIS 16E ARRDT',
        'PARIS 17E ARRDT',
        'PARIS 18E ARRDT',
        'PARIS 19E ARRDT',
        'PARIS 20E ARRDT',
        'BOIS DE BOULOGNE',
        'BOIS DE VINCENNES',
        'HAUTS-DE-SEINE',
        'SEINE-SAINT-DENIS',
        'VAL-DE-MARNE']
    road_dict = dict()
    km_dict = dict()
    km_pm_dict = dict()
    nb_lieu_dict = dict()
    info_dict = dict()
    for arr in arr_list:
        if (not df_graph.loc[df_graph['arrond'] == arr].empty)&(df_graph['lieu'].loc[df_graph['arrond'] == arr].count()>4):
            print('df_graph',str(df_graph.loc[df_graph['arrond'] == arr].count()))
            print('Pour l\'arrondissement ' + arr + ' : \n')
            df_graph.loc[df_graph['arrond'] == arr].to_excel('data/export_df_graph.xlsx')

            global d  # Notre table de données
            global SIZE
            matrix()

            distances_gpu = pd.read_pickle(
                PICKLE_FILE_1)  # .drop(['Unnamed: 0'], axis=1).rename(columns={'Unnamed: 0.1':'lieu'})
            distances_gpu['lieu'] = distances_gpu.columns
            distances_gpu.index = distances_gpu['lieu']
            distances_gpu.drop(['lieu'], axis=1, inplace=True)
            SIZE = len(distances_gpu.index)
            d = distances_gpu.iloc[:SIZE, :SIZE].copy()

            list_pm, km = poids_min(d)
            print('Au départ le chemin de poids mininum est : ', km)
            list_imp, list_imp_sec = impair(list_pm)
            list_imp_pm, km_imp = poids_min_impair(list_imp)
            union = liaison(list_pm, list_imp_sec)

            chemin = Graph(union)
            for i in range(len(union)):
                chemin.add_edge(i)
            print(chemin)

            print('le chemin de poids nominal était de : ', int(km), 'km\n[fin]\n\n')
            road_dict.update({arr: chemin.road})
            km_dict.update({arr: chemin.km_final})
            km_pm_dict.update({arr: km})
            nb_lieu_dict.update({arr: len(chemin.road)})
        else:
            continue
    info_dict.update({'road': road_dict})
    info_dict.update({'km':km_dict})
    info_dict.update({'km_pm': km_pm_dict})
    info_dict.update({'nb_lieu': nb_lieu_dict})
    road_dict_df = dict()
    for arr in arr_list:
        try:
            road_dict_df.update({arr: info_dict['road'][arr]})
        except KeyError as e:
            print(e)
            pass
    road_df = pd.DataFrame.from_dict(road_dict_df, orient='index')
    road_df.transpose().to_excel('data/chemin_par_' + col + '_arrondissement_' + str(q_h) + '_' + str(q_b) + '.xlsx')
    write_to_pickle('data/chemin_'+ col +'_' + str(q_h) + '_' + str(q_b) + '.dat', info_dict)

def chemin(q_h=0.95, q_b=0.05):
    for col in ['soin', 'ratio', 'remarquable']:
        chemin_note = pd.read_pickle('data/chemin_' + col + '_' + str(q_h) + '_' + str(q_b) + '.dat')
        road_dict_df = dict()
        for arr in arr_list:
            try:
                road_dict_df.update({arr: chemin_note['road'][arr]})
            except KeyError as e:
                print(e)
                pass
        road_df = pd.DataFrame.from_dict(road_dict_df, orient='index')
        road_df.transpose().to_excel(
            'data/chemin_par_' + col + '_arrondissement_' + str(q_h) + '_' + str(q_b) + '.xlsx')

def main():
    christofides(col='soin')
    christofides(col='ratio')
    christofides(col='remarquable')
    chemin()

if __name__ == '__main__':
    main()
