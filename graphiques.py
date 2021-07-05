import pandas as pd
import pandasql as ps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_filtre = pd.read_excel('data_filtre.xlsx')





for x,y in [[0.8, 0.2],[0.85,0.15],[0.90,0.10],[0.95,0.05],[0.98,0.02],[0.99,0.01]]:

    q_h = x
    q_b = y
    mean = pd.DataFrame()
    std = pd.DataFrame()
    data_filtre_moy_qut = pd.DataFrame({"Quantite": pd.Series([], dtype='int'),
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
            data_filtre_moy_qut = data_filtre_moy_qut.append({'libelle_francais': arr, 'age': age, 'Quantite':
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
        ['libelle_francais', 'age', 'Quantite', 'avg_h', 'std_h', 'qut_h_h', 'qut_h_b', 'avg_c', 'std_c', 'qut_c_h',
         'qut_c_b']].sort_values(by=['libelle_francais', 'age']).reset_index()

    print(data_filtre_moy_qut.info())
    q8 = """SELECT  data_filtre_moy_qut.libelle_francais,
                    data_filtre_moy_qut.age,
                    data_filtre_moy_qut.Quantite,
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
    actions['sante'] = ''
    actions['soin'] = ''

    load = actions.copy()
    for lib in list(load.libelle_francais.unique()):
        load.loc[actions['libelle_francais'] == lib,'nb_arbre_meme_libel'] = load.libelle_francais.loc[actions['libelle_francais'] == lib].count()
    for arr in list(load.arrondissement.unique()):
        load.loc[actions['arrondissement'] == arr,'nb_arbre_meme_arr'] = load.arrondissement.loc[actions['arrondissement'] == arr].count()

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


    for arr in list(load.arrondissement.unique()):
        load.loc[(actions['arrondissement'] == arr)&(actions['soin'] == 'à surveiller'),'nb_arbre_surveiller_arr'] = load.arrondissement.loc[actions['arrondissement'] == arr].count()

    actions = load.copy()

    # for age in list(actions.unique()):
    sns.set_theme(style="whitegrid")

    #df_age_quant_sante = actions[['avg_h','sante', 'avg_c', 'Quantite']].loc[actions['age'] == 'A']# age instead of 'J'

    f, ax = plt.subplots(figsize=(8.5, 8.5))
    sns.despine(f, left=True, bottom=True)
    plt.title(str(actions['libelle_francais'].loc[actions['soin']=='à surveiller'].count()) + ' arbres à surveiller\n'
              +str(actions['libelle_francais'].loc[actions['soin']=='normal'].count()) + ' arbres sains',
              fontdict={'fontsize':21, 'fontweight': 'bold'})
    plt.ylabel('La circonférence moyenne')
    plt.xlabel('La hauteur moyenne')
    #plt.xscale('Symlog')
    #plt.yscale('Symlog')
    #clarity_ranking = ['M','A','JA','P','J']
    sns.scatterplot(x="value_moy_c", y='value_moy_h',
                    hue="sante", style_order=['J','P','JA','A','M'],
                    size='Quantite',sizes=(50, 200),
                    style='age', size_order=(actions['nb_arbre_meme_libel'].min(),actions['nb_arbre_meme_libel'].min()),
                    data=actions[['value_moy_h','arrondissement','value_moy_c','age','sante','soin','Quantite']], ci="sd",ax=ax)
    f.savefig('img/arbres_surveillance_lineaire_' + str(q_h) +'_'+ str(q_b)+ '.png')
    plt.clf()
    plt.close('all')

    # for age in list(actions.unique()):
    sns.set_theme(style="whitegrid")

    #df_age_quant_sante = actions[['avg_h','sante', 'avg_c', 'Quantite']].loc[actions['age'] == 'A']# age instead of 'J'

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
                    size='Quantite',sizes=(50, 200),
                    style='age', size_order=(actions['nb_arbre_meme_libel'].min(),actions['nb_arbre_meme_libel'].min()),
                    data=actions[['value_moy_h','arrondissement','value_moy_c','age','sante','soin','Quantite']], ci="sd",ax=ax)
    f.savefig('img/arbre_norm_surv_par_age_' + str(q_h) +'_'+ str(q_b)+ '.png')
    plt.clf()
    plt.close('all')

    # for age in list(actions.unique()):
    sns.set_theme(style="whitegrid")

    #df_age_quant_sante = actions[['avg_h','sante', 'avg_c', 'Quantite']].loc[actions['age'] == 'A']# age instead of 'J'
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
                    size='Quantite',sizes=(50, 200),
                    style='age',
                     size_order=(actions['nb_arbre_meme_libel'].min(),actions['nb_arbre_meme_libel'].min()),
                    data=actions[['value_moy_h','arrondissement','value_moy_c','age','sante','soin','Quantite']].loc[actions['soin']=='à surveiller'], ci='sd',ax=ax)
    f.savefig('img/data_arbre_a_surveiller_par_arrondissement_q_h_' + str(q_h) +'_'+ str(q_b)+ '.png')
    plt.clf()
    plt.close('all')

    actions.to_excel('actions_' + str(q_h) + '_' + str(q_b) + '.xlsx')

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