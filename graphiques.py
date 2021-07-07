import folium
import pandas as pd
import pandasql as ps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import christofides
data_filtre = pd.read_excel('data/data_filtre.xlsx')
data = pd.read_csv('data.csv', sep=';')




for x,y in [[0.95,0.05]]:#,[0.8, 0.2],[0.85,0.15],[0.90,0.10],[0.98,0.02],[0.99,0.01]]:
    q_h = x
    q_b = y
    mean = pd.DataFrame()
    std = pd.DataFrame()
    data_filtre_moy_qut = pd.DataFrame({"quantite_var_par_age": pd.Series([], dtype='int'),
         "age": pd.Series([], dtype= 'str'),
         "avg_c": pd.Series([], dtype= "float"),
         "avg_h": pd.Series([], dtype= "float"),
         "libelle_francais": pd.Series([], dtype='str'),
        "qut_c_b": pd.Series([], dtype="float"),
        "qut_c_h": pd.Series([], dtype="float"),
        "qut_h_b": pd.Series([], dtype="float"),
        "qut_h_h": pd.Series([], dtype="float"),
        "std_c": pd.Series([], dtype="float"),
        "std_h": pd.Series([], dtype="float")})

    for arr in list(data_filtre['libelle_francais'].unique()):
        for age in list(data_filtre['stade_developpement'].unique()):
            mean = data_filtre.loc[
                (data_filtre['libelle_francais'] == arr) & (data_filtre['stade_developpement'] == age)].mean()
            std = data_filtre.loc[
                (data_filtre['libelle_francais'] == arr) & (data_filtre['stade_developpement'] == age)].std()
            qut_h = data_filtre.loc[
                (data_filtre['libelle_francais'] == arr) & (data_filtre['stade_developpement'] == age)].quantile(q=q_h)
            qut_b = data_filtre.loc[
                (data_filtre['libelle_francais'] == arr) & (data_filtre['stade_developpement'] == age)].quantile(q=q_b)
            data_filtre_moy_qut = data_filtre_moy_qut.append({'libelle_francais': arr, 'age': age, 'quantite_var_par_age':
                int(data_filtre['type_emplacement'].loc[(data_filtre['libelle_francais'] == arr) &
                                                        (data_filtre['stade_developpement'] == age)].count()),

                                                              'avg_h': float(mean['hauteur_m']), 'std_h': std['hauteur_m'],
                                                              'qut_h_h': qut_h['hauteur_m'], 'qut_h_b': qut_b['hauteur_m'],
                                                              'avg_c': float(mean['circonference_cm']),
                                                              'std_c': std['circonference_cm'],
                                                              'qut_c_h': qut_h['circonference_cm'],
                                                              'qut_c_b': qut_b['circonference_cm']},
                                                             ignore_index=True)
    data_filtre_moy_qut[
        ['libelle_francais', 'age', 'quantite_var_par_age', 'avg_h', 'std_h', 'qut_h_h', 'qut_h_b', 'avg_c', 'std_c', 'qut_c_h',
         'qut_c_b']].sort_values(by=['libelle_francais', 'age']).reset_index()

    print(data_filtre_moy_qut.info())
    q8 = """SELECT  data_filtre_moy_qut.libelle_francais,
                    data_filtre_moy_qut.age,
                    data_filtre_moy_qut.quantite_var_par_age,
                    data_filtre_moy_qut.avg_h,
                    data_filtre_moy_qut.qut_h_h,
                    data_filtre_moy_qut.qut_h_b,
                    data_filtre_moy_qut.std_h,
                    data_filtre_moy_qut.avg_c,
                    data_filtre_moy_qut.qut_c_h,
                    data_filtre_moy_qut.qut_c_b,
                    data_filtre_moy_qut.std_c,
                    data_filtre.type_emplacement,
                    data_filtre.domanialite,
                    data_filtre.arrondissement,
                    data_filtre.complement_addresse,
                    data_filtre.lieu,
                    data_filtre.id_emplacement,
                    data_filtre.genre,
                    data_filtre.espece,
                    data_filtre.variete,
                    data_filtre.hauteur_m,
                    data_filtre.stade_developpement,
                    data_filtre.circonference_cm,
                    data_filtre.remarquable,
                    data_filtre.sta_dev_num,
                    data_filtre.geo_point_2d_a as lat,
                    data_filtre.geo_point_2d_b as lon
                    FROM data_filtre 
                    JOIN data_filtre_moy_qut 
                    ON data_filtre_moy_qut.libelle_francais = data_filtre.libelle_francais 
                    AND data_filtre.stade_developpement = data_filtre_moy_qut.age """

    actions = ps.sqldf(q8, locals())

    ## Dépose d'un marqueur de soin concernant la santé de l'arbre.
    actions['sante'] = ''
    actions['soin'] = ''

    load = actions.copy()
    for lib in list(load.libelle_francais.unique()):
        load.loc[actions['libelle_francais'] == lib, 'nb_arbre_meme_libel'] = load.libelle_francais.loc[actions['libelle_francais'] == lib].count()
    for arr in list(load.arrondissement.unique()):
        load.loc[actions['arrondissement'] == arr, 'nb_arbre_meme_arr'] = load.arrondissement.loc[actions['arrondissement'] == arr].count()
    for arr in list(load.arrondissement.unique()):
        for soin in ['à surveiller']:
            load.loc[actions['arrondissement'] == arr, 'nb_surv_arr'] = load.arrondissement.loc[(actions['arrondissement'] == arr) & (actions['soin'] == soin)].count()
    for lieu in list(load.lieu.unique()):
        for soin in ['à surveiller']:
            load.loc[actions['lieu'] == lieu, 'nb_surv_lieu'] = load.lieu.loc[(actions['lieu'] == lieu) & (actions['soin'] == soin)].count()

    load.loc[(load['hauteur_m'] <= load['qut_h_b'])|(load['circonference_cm'] <= load['qut_c_b']), 'sante'] = 'au-dessous'
    load.loc[(load['hauteur_m'] <= load['qut_h_b'])|(load['circonference_cm'] <= load['qut_c_b']), 'soin'] = 'à surveiller'
    load.loc[(load['hauteur_m'] <= load['qut_h_b'])|(load['circonference_cm'] <= load['qut_c_b']), 'value_moy_c'] = load['qut_c_b'].loc[(load['hauteur_m'] <= load['qut_h_b'])&(load['circonference_cm'] <= load['qut_c_b'])]
    load.loc[(load['hauteur_m'] <= load['qut_h_b'])|(load['circonference_cm'] <= load['qut_c_b']), 'value_moy_h'] = load['qut_h_b'].loc[(load['hauteur_m'] <= load['qut_h_b'])&(load['circonference_cm'] <= load['qut_c_b'])]

    load.loc[(load['hauteur_m'] >= load['qut_h_h'])|(load['circonference_cm'] >= load['qut_c_h']), 'sante'] = 'au-dessus'
    load.loc[(load['hauteur_m'] >= load['qut_h_h'])|(load['circonference_cm'] >= load['qut_c_h']), 'soin'] = 'à surveiller'
    load.loc[(load['hauteur_m'] >= load['qut_h_h'])|(load['circonference_cm'] >= load['qut_c_h']), 'value_moy_c'] = load['qut_c_h'].loc[(load['hauteur_m'] >= load['qut_h_h'])&(load['circonference_cm'] >= load['qut_c_h'])]
    load.loc[(load['hauteur_m'] >= load['qut_h_h'])|(load['circonference_cm'] >= load['qut_c_h']), 'value_moy_h'] = load['qut_h_h'].loc[(load['hauteur_m'] >= load['qut_h_h'])&(load['circonference_cm'] >= load['qut_c_h'])]

    load.loc[((load['hauteur_m'] < load['qut_h_h'])&(load['circonference_cm'] < load['qut_c_h']))
             &((load['hauteur_m'] > load['qut_h_b'])&(load['circonference_cm'] > load['qut_c_b'])), 'sante'] = 'normal'
    load.loc[((load['hauteur_m'] < load['qut_h_h'])&(load['circonference_cm'] < load['qut_c_h']))
             &((load['hauteur_m'] > load['qut_h_b'])&(load['circonference_cm'] > load['qut_c_b'])), 'soin'] = 'normal'
    load.loc[((load['hauteur_m'] < load['qut_h_h'])&(load['circonference_cm'] < load['qut_c_h']))
             &((load['hauteur_m'] > load['qut_h_b'])&(load['circonference_cm'] > load['qut_c_b'])), 'value_moy_c'] = load['avg_c'].loc[((load['hauteur_m'] < load['qut_h_h'])|(load['circonference_cm'] < load['qut_c_h']))]
    load.loc[((load['hauteur_m'] < load['qut_h_h'])&(load['circonference_cm'] < load['qut_c_h']))
             &((load['hauteur_m'] > load['qut_h_b'])&(load['circonference_cm'] > load['qut_c_b'])), 'value_moy_h'] = load['avg_h'].loc[((load['hauteur_m'] < load['qut_h_h'])|(load['circonference_cm'] < load['qut_c_h']))]

    #
    for lib in list(load.libelle_francais.unique()):
        load.loc[actions['libelle_francais'] == lib, 'nb_arbre_meme_libel'] = load.libelle_francais.loc[actions['libelle_francais'] == lib].count()
    for arr in list(load.arrondissement.unique()):
        load.loc[actions['arrondissement'] == arr, 'nb_arbre_meme_arr'] = load.arrondissement.loc[
            actions['arrondissement'] == arr].count()
    for lib in list(load.libelle_francais.unique()):
        for arr in list(load.arrondissement.unique()):
            load.loc[(actions['arrondissement'] == arr) & (actions['libelle_francais'] == lib), 'nb_arbre_meme_libel_arr'] = \
            load.arrondissement.loc[(actions['arrondissement'] == arr) & (actions['libelle_francais'] == lib)].count()

    actions = load.copy()

    # for age in list(actions.unique()):
    sns.set_theme(style="whitegrid")

    #df_age_quant_sante = actions[['avg_h','sante', 'avg_c', 'quantite_var_par_age']].loc[actions['age'] == 'A']# age instead of 'J'

    f, ax = plt.subplots(figsize=(8.5, 8.5))
    sns.despine(f, left=True, bottom=True)
    plt.title(str(actions['libelle_francais'].loc[actions['soin']=='à surveiller'].count()) + ' arbres à surveiller\n'
              +str(actions['libelle_francais'].loc[actions['soin']=='normal'].count()) + ' arbres sains',
              fontdict={'fontsize':21, 'fontweight': 'bold'})
    plt.ylabel('La circonférence moyenne')
    plt.xlabel('La hauteur moyenne')

    sns.scatterplot(x="value_moy_c", y='value_moy_h',
                    hue="sante", style_order=['J','P','JA','A','M'],
                    size='quantite_var_par_age',sizes=(50, 200),
                    style='age', size_order=(actions['nb_arbre_meme_libel'].min(),actions['nb_arbre_meme_libel'].min()),
                    data=actions[['value_moy_h','arrondissement','value_moy_c','age','sante','soin','quantite_var_par_age']], ci="sd",ax=ax)
    f.savefig('img/arbres_surveillance_lineaire_' + str(q_h) +'_'+ str(q_b)+ '.png')
    plt.clf()
    plt.close('all')

    # for age in list(actions.unique()):
    sns.set_theme(style="whitegrid")

    #df_age_quant_sante = actions[['avg_h','sante', 'avg_c', 'quantite_var_par_age']].loc[actions['age'] == 'A']# age instead of 'J'

    f, ax = plt.subplots(figsize=(8.5, 8.5))
    sns.despine(f, left=True, bottom=True)
    plt.title(str(actions['libelle_francais'].loc[actions['soin']=='à surveiller'].count()) + ' arbres à surveiller\n'
              +str(actions['libelle_francais'].loc[actions['soin']=='normal'].count()) + ' arbres sains',
              fontdict={'fontsize':21, 'fontweight': 'bold'})
    plt.ylabel('La circonférence moyenne')
    plt.xlabel('La hauteur moyenne')
    plt.xscale('Symlog')
    plt.yscale('Linear')
    #clarity_ranking = ['M','A','JA','P';'J']
    sns.scatterplot(x="value_moy_c", y='value_moy_h',
                    hue="soin", style_order=['J','P','JA','A','M'],
                    size='quantite_var_par_age',sizes=(50, 200),
                    style='age', size_order=(actions['nb_arbre_meme_libel'].min(),actions['nb_arbre_meme_libel'].min()),
                    data=actions[['value_moy_h','arrondissement','value_moy_c','age','sante','soin','quantite_var_par_age']], ci="sd",ax=ax)
    f.savefig('img/arbre_norm_surv_par_age_' + str(q_h) +'_'+ str(q_b)+ '.png')
    plt.clf()
    plt.close('all')

    # for age in list(actions.unique()):
    sns.set_theme(style="whitegrid")

    #df_age_quant_sante = actions[['avg_h','sante', 'avg_c', 'quantite_var_par_age']].loc[actions['age'] == 'A']# age instead of 'J'
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
    f, ax = plt.subplots(figsize=(12.5, 15.5))
    sns.despine(f, left=True, bottom=True)
    plt.title('Les arbres à surveiller par arrondissement',
              fontdict={'fontsize':21, 'fontweight': 'bold'})
    plt.ylabel('La circonférence moyenne')
    plt.xlabel('La hauteur moyenne')
    plt.xscale('Symlog')
    plt.yscale('Linear')#clarity_ranking = ['M','A','JA','P','J']
    sns.scatterplot(x="value_moy_c", y='value_moy_h',
                    hue_order=arr_list,
                    hue="arrondissement", style_order=['J','P','JA','A','M'],
                    size='quantite_var_par_age',sizes=(50, 200),
                    style='age',
                     size_order=(actions['nb_arbre_meme_libel'].min(),actions['nb_arbre_meme_libel'].min()),
                    data=actions[['value_moy_h','arrondissement','value_moy_c','age','sante','soin','quantite_var_par_age']].loc[actions['soin']=='à surveiller'], ci='sd',ax=ax)
    f.savefig('img/data_arbre_a_surveiller_par_arrondissement_q_h_' + str(q_h) +'_'+ str(q_b)+ '.png')
    plt.clf()
    plt.close('all')

    actions.to_excel('data/actions_' + str(q_h) + '_' + str(q_b) + '.xlsx')
### Creation du nombre d'arbre par arrondissement
    q9 = """SELECT actions.arrondissement as arr, 
                    COUNT(*) as total
                    FROM actions
           GROUP BY actions.arrondissement
    """

    actions_hist = ps.sqldf(q9, locals())

    q10 = """SELECT actions.arrondissement as arr, 
                    COUNT(soin) as surveiller
                    FROM actions WHERE soin LIKE '%surveiller'
           GROUP BY actions.arrondissement
    """

    actions_hist_soin = ps.sqldf(q10, locals())

    q11 = """SELECT actions_hist.arr as arr, 
                    actions_hist.total,
                    actions_hist_soin.surveiller 
                    FROM actions_hist JOIN actions_hist_soin ON actions_hist.arr = actions_hist_soin.arr
    
    
    """

    actions_hist_merge = ps.sqldf(q11, locals())

    actions_hist_merge.to_excel('data/quant_surveiller_arrondissement_q_h_' + str(q_h) + '_' + str(q_b) + '.xlsx')

    data_hist_chart = actions_hist_merge.copy()

    sns.set_theme(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(16, 15))

    # Load the example car crash dataset
    crashes = data_hist_chart
    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(order=arr_list,x="total", y="arr", data=crashes,
                label="Normaux", color="b")

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(order=arr_list,x="surveiller", y="arr", data=crashes,
                label="à Surveiller", color="b")

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 15000), ylabel="",
           xlabel="Les arbres par arrondissement")
    sns_plot = sns.despine(left=True, bottom=True)
    f.savefig('img/arbre_a_surveiller_par_arrondissement_q_h_' + str(q_h) +'_'+ str(q_b)+ '.png')
    plt.clf()
    plt.close('all')

    ### Analyse des arbres les plus fréquents par arrondissement

    freq_libel_arr = actions[['arrondissement', 'libelle_francais', 'nb_arbre_meme_libel_arr']].copy()

    freq_libel_arr = freq_libel_arr[~freq_libel_arr.duplicated()].sort_values(by='arrondissement')

    freq = pd.DataFrame()
    for arr in list(freq_libel_arr['arrondissement'].unique()):
        freq = freq.append(freq_libel_arr.loc[freq_libel_arr['arrondissement'] == arr]
                           .sort_values(by='nb_arbre_meme_libel_arr').iloc[-5:])

    freq_libel_arr.to_excel('draft/data_freq_libel_arr.xlsx')

    f, ax = plt.subplots(figsize=(60, 60))
    A = np.array(list(freq['nb_arbre_meme_libel_arr'].iloc[0::5]))
    B = np.array(list(freq['nb_arbre_meme_libel_arr'].iloc[1::5]))
    C = np.array(list(freq['nb_arbre_meme_libel_arr'].iloc[2::5]))
    D = np.array(list(freq['nb_arbre_meme_libel_arr'].iloc[3::5]))
    E = np.array(list(freq['nb_arbre_meme_libel_arr'].iloc[4::5]))
    Pos = range(25)
    plt.bar(Pos, A)
    plt.bar(Pos, B, bottom=A)
    plt.bar(Pos, C, bottom=A + B)
    plt.bar(Pos, D, bottom=A + B + C)
    plt.bar(Pos, E, bottom=A + B + C + D)
    plt.xticks(Pos, arr_list)
    plt.title('La part des 5 libellés les plus fréquents par arrondissement',
              fontdict={'fontsize': 60, 'fontweight': 'bold'})
    plt.ylabel('La circonférence moyenne')
    plt.xlabel('La hauteur moyenne')
    f.savefig('img/part_des_5_libels_freq_par_arrondissement_' + str(q_h) + '_' + str(q_b) + '.png')
    plt.close('all')

    ### Analyse des emplacements des arbres remarquables

    remarkable = actions.loc[actions.remarquable == 1].copy()

    dom_rem_df = pd.DataFrame(columns=['dom', 'count'])
    for dom in list(remarkable.domanialite.unique()):
        # print(dom,remarkable['domanialite'].loc[remarkable['domanialite']==dom].count())
        dom_rem_df = dom_rem_df.append(
            {'dom': dom, 'count': remarkable['domanialite'].loc[remarkable['domanialite'] == dom].count()},
            ignore_index=True)

    f, ax = plt.subplots(figsize=(18, 15))
    sns.barplot(data=dom_rem_df, order=sorted(list(remarkable['domanialite'].unique())), y="dom", x="count",
                orient='h')  # , multiple="stack")
    plt.title('La répartition des arbres remarquables par domanialité', fontdict={'fontsize': 30, 'fontweight': 'bold'})
    plt.ylabel('domanialités')
    plt.xlabel('nb arbres remarquables')
    f.savefig('img/remarquable_par_domanialite_' + str(q_h) + '_' + str(q_b) + '.png')
    plt.close('all')

    arr_rem_df = pd.DataFrame(columns=['arr', 'count'])
    for arr in list(remarkable.arrondissement.unique()):
        # print(dom,remarkable['arrondissement'].loc[remarkable['arrondissement']==arr].count())
        arr_rem_df = arr_rem_df.append(
            {'arr': arr, 'count': remarkable['arrondissement'].loc[remarkable['arrondissement'] == arr].count()},
            ignore_index=True)

    f, ax = plt.subplots(figsize=(18, 15))
    sns.barplot(data=arr_rem_df, order=arr_list, y="arr", x="count", orient='h')  # , multiple="stack")
    plt.title('La répartition des arbres remarquables par arrondissement',
              fontdict={'fontsize': 30, 'fontweight': 'bold'})
    plt.ylabel('arrondissemeent')
    plt.xlabel('nb arbres remarquables')
    f.savefig('img/remarquable_par_arrondissement_' + str(q_h) + '_' + str(q_b) + '.png')
    plt.close('all')

    ### Analyse du nombre d'arbre par arrondissement et par stade de développement.

    qa9 = """SELECT actions.arrondissement as arr, 
                    actions.age,
                    COUNT(*) as total
                    FROM actions
           GROUP BY actions.arrondissement, actions.age
    """

    actions_hist_stadev = ps.sqldf(qa9, locals())

    actions_hist_stadev

    qa10 = """SELECT actions.arrondissement as arr, 
                    actions.age,
                    COUNT(soin) as surveiller
                    FROM actions WHERE soin LIKE '%surveiller'
           GROUP BY actions.arrondissement, actions.age
    """

    actions_hist_stadev_soin = ps.sqldf(qa10, locals())

    actions_hist_stadev_soin

    qa11 = """SELECT actions_hist_stadev.arr as arr, 
                    actions_hist_stadev.age,
                    actions_hist_stadev.total,
                    actions_hist_stadev_soin.surveiller 
                    FROM actions_hist_stadev JOIN actions_hist_stadev_soin ON actions_hist_stadev.arr = actions_hist_stadev_soin.arr AND
                    actions_hist_stadev.age = actions_hist_stadev_soin.age


    """

    actions_hist_stadev_merge = ps.sqldf(qa11, locals())

    actions_hist_stadev_merge

    data_hist__stdev_chart = actions_hist_stadev_merge.copy()

    sns.set_theme(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(8, 15))

    # Load the example car crash dataset
    crashes = data_hist__stdev_chart
    # Plot the total crashes
    sns.set_color_codes("muted")
    sns.barplot(order=arr_list, hue_order=['J', 'P', 'JA', 'A', 'M'], x="total", y="arr", hue="age", data=crashes,
                color="g")

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 15000), ylabel="",
           xlabel="Les arbres par arrondissement, par âge")
    sns_plot = sns.despine(left=True, bottom=True)
    f.savefig('img/nombre_arbre_par_arrondissement_par_stade_de_dev_q_h_' + str(q_h) +'_'+ str(q_b)+ '.png')
    plt.clf()
    plt.close('all')


    f = plt.figure(figsize=(20, 20))
    ax = sns.violinplot(order=['J', 'P', 'JA', 'A', 'M'], x="age", hue="remarquable", y="quantite_var_par_age",
                        data=actions, inner=None, scale='width')
    f.savefig('img/repartition_des_varietes_par_age_par_remarquable.png')
    plt.close('all')
    q12 = """SELECT actions.lieu as lieu, 
                    actions.arrondissement,
                    COUNT(*) as total
                    FROM actions
           GROUP BY actions.lieu
    """
    actions_hist_lieu = ps.sqldf(q12, locals())

    q13 = """SELECT actions.lieu as lieu, 
                    actions.arrondissement,
                    COUNT(soin) as surveiller
                    FROM actions WHERE soin LIKE '%surveiller'
           GROUP BY actions.lieu
    """
    actions_hist_lieu_soin = ps.sqldf(q13, locals())
    q14 = """SELECT actions_hist_lieu_soin.lieu as lieu, 
                    actions_hist_lieu.arrondissement,
                    actions_hist_lieu.total,
                    actions_hist_lieu_soin.surveiller 
                    FROM actions_hist_lieu JOIN actions_hist_lieu_soin ON actions_hist_lieu.lieu = actions_hist_lieu_soin.lieu


    """
    actions_hist_lieu_merge = ps.sqldf(q14, locals())
    actions_hist_lieu_merge.to_excel('data/quant_surveiller_lieu_q_h_'+ str(q_h) + '_' + str(q_b) + '.xlsx')
    f = plt.figure(figsize=(10, 10))
    ax = actions_hist_lieu_merge['surveiller'].plot()
    plt.title('Nombre d\'arbre à surveiller à chaque lieu', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.xlabel('Lieu')
    plt.ylabel('arbres à surveiller')
    plt.savefig('img/plot_surveiller_lieu_q_h_'+ str(q_h) + '_' + str(q_b) + '.png')
    plt.close('all')
    christofides.christofides(x, y)
    chemin_note = pd.read_pickle('chemin.dat_' + str(q_h) + '_' + str(q_b))
    road_dict_df = dict()
    for arr in arr_list:
        road_dict_df.update({arr: chemin_note['road'][arr]})
    road_df = pd.DataFrame.from_dict(road_dict_df, orient='index')
    road_df.transpose().to_excel('data/chemin_par_arrondissement_' + str(q_h) + '_' + str(q_b) + '.xlsx')

    f = plt.figure(figsize=(20, 20))
    ax = sns.violinplot(order=['J', 'P', 'JA', 'A', 'M'], x="age", y="hauteur_m", data=actions, inner=None,
                        scale='width')
    f.savefig('img/repartition_des_varietes_par_age_par_remarquable.png')
    plt.close('all')
    f = plt.figure(figsize=(20, 20))
    ax = sns.violinplot(order=['J', 'P', 'JA', 'A', 'M'], x="age", y="circonference_cm", data=actions, inner=None,
                        scale='width')
    f.savefig('img/repartition_des_varietes_par_age_par_remarquable.png')
    plt.close('all')

    actions_simple = pd.read_excel('data/actions_' + str(q_h) + '_' + str(q_b) + '.xlsx')
    actions_simple = actions_simple.rename(columns={'lat': 'geo_point_2d_a', 'lon': 'geo_point_2d_b'})
    action_df_simple = actions_simple[['type_emplacement', 'domanialite', 'arrondissement',
                                       'complement_addresse', 'lieu', 'id_emplacement', 'libelle_francais',
                                       'genre', 'espece', 'variete', 'circonference_cm', 'hauteur_m',
                                       'stade_developpement', 'remarquable', 'geo_point_2d_a',
                                       'geo_point_2d_b', 'soin']].copy()
    data_new = data.loc[((data["hauteur_m"] >= 21) | (data["hauteur_m"] <= 0)) \
                        | ((data["circonference_cm"] >= 255) | (data["circonference_cm"] <= 0))].copy()
    data_new['soin'] = 'à vérifier'
    data_end = action_df_simple.append(data_new).copy()
    #data_end = data_end.drop(['sta_dev_num', 'couleurs'], axis=1)
    data_end.to_excel('data/new_data_end_' + str(q_h) + '_' + str(q_b) + '.xlsx')
    ## Faire la carte
    data_map = pd.read_excel('data/new_data_end_' + str(q_h) + '_' + str(q_b) +'.xlsx')

    poi_list = ['orange', 'blue', 'red']
    poi_icon_list = ['heart', 'ok-sign', 'question-sign']
    poi_dict = dict()
    for it, val in enumerate(data_end['soin'].unique()):
        poi_dict.update({val: poi_list[it]})
    poi_icon_dict = dict()
    for it, val in enumerate(data_end['soin'].unique()):
        poi_icon_dict.update({val: poi_icon_list[it]})
    poi_icon_dict

    data_map['poi'] = data_map['soin'].map(poi_dict)
    data_map['poi_icon'] = data_map['soin'].map(poi_icon_dict)

    m = folium.Map(location=[48.857722, 2.321031], zoom_start=12, tiles="Stamen Terrain")
    for i in range(20000):
        folium.Marker(
            location=[data_map['geo_point_2d_a'].iloc[i], data_map['geo_point_2d_b'].iloc[i]],
            popup=str(data_map['genre'].iloc[i]) + ', ' + str(data_map['espece'].iloc[i]) + ', ' + str(
                data_map['variete'].iloc[i]) + ', ' +
                  str(data_map['libelle_francais'].iloc[i]) + ', haut. : ' + str(float(data_map['hauteur_m'].iloc[i])) +
                  'm, circ. : ' + str(float(data_map['circonference_cm'].iloc[i])) + 'cm, action à prendre : ' +
                  data_map['soin'].iloc[i],
            icon=folium.Icon(icon=data_map['poi_icon'].iloc[i], color=data_map['poi'].iloc[i]),
        ).add_to(m)

    m.save("data/map/index_"+ str(q_h) + '_'+ str(q_b) + ".html")



