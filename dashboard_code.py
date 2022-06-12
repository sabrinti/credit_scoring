#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import st_aggrid
import joblib
from joblib import load , dump
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import DistanceMetric
import math
import base64
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st
from collections import defaultdict
from PIL import Image 
import lightgbm as lgb
import requests
import json
import plotly.graph_objects as go
import shap





def predict_credit_default_client(model, id_client, data, threshold):
    ID = int(id_client)
    X = data[data['SK_ID_CURR'] == ID]
    X = X.drop(['SK_ID_CURR'], axis=1)
    proba_default = model.predict_proba(X)[:, 1]
    if proba_default >= threshold:
            prediction = "Client défaillant: prêt refusé"
    else:
            prediction = "Client non défaillant:  prêt accordé"
    return proba_default, prediction

    
# Chargement des tables du modèle
def load_data_model():
    lgbm=joblib.load('lgbm_hp_seuil.joblib')
    data= pd.read_csv('data_reduced.csv')
    df_test_sample = pd.read_csv('data_reduced.csv', index_col=0)
    all_id_client = df_test_sample.index.sort_values()
    df_int = pd.read_csv('df_interprete_pl.csv')
    nearest_n = pickle.load(open('NearestNeighborsModel.pkl', 'rb'))
    std = pickle.load(open('StandardScaler.pkl', 'rb'))
    data_train_id_target = pd.read_csv('data_id_target_reduced.csv')
    data_with_target = pd.read_csv('data_train_id_target_reduced.csv')
    feature_important_data = ['SK_ID_CURR',
                          'CODE_GENDER',
                          'ANNUITY_CREDIT_PERCENT_INCOME',
                          'AMT_ANNUITY',
                          'AMT_INCOME_TOTAL',
                          'AGE',
                          'DAYS_EMPLOYED_PERCENT',
                          'CREDIT_REFUND_TIME',
                             'CNT_CHILDREN']
    
    return lgbm, data, all_id_client, df_int, feature_important_data, std, nearest_n, data_train_id_target, data_with_target








def identite_client(df_int, id):
        data_client = df_int[df_int['Id client'] == int(id)]
        return data_client


def load_age_population(df_int):
        data_age = df_int["Age client (ans)"]
        return data_age

def load_income_population(df_int):
        df_income = pd.DataFrame(df_int["Revenus globaux"])
        df_income = df_income.loc[df_income['Revenus globaux'] < 300000, :]
        return df_income
    

         
    
def is_valid_client(id_client, all_client):
    if id_client not in all_client:
        return "ko"
    else:
        return "ok"
    
def predict_client(id_client):
    valid = is_valid_client(id_client, all_id_client)
    if valid == "ko":
        client_non_repertorie =  '<p class="p-style-red"><b>Ce client n\'est pas répertorié</b></p>'
        st.markdown(client_non_repertorie, unsafe_allow_html=True)    
        return "ko"

    else:
         st.markdown("##### ")  

    if valid == "ok":       
    
        API_URL = 'https://scoring-sabrine.herokuapp.com/'
        data_mean = API_URL + "/predict?id=" + str(idclient)
         #print(data_mean)
        response = requests.get(data_mean)
        content = json.loads(response.content.decode('utf-8'))
    
        confidence = content[0]['confidence']
        prediction = content[0]['prediction']
        st.write(f'Le client reçoit son crédit : {prediction}')
        #st.write(f'#### Score de prediction : {confidence}')
        #st.write(f'Probabilité de défauts de remboursement: {confidence}')
        fig_gauge = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = float(f'{confidence}'),
        mode = "gauge+number+delta",
        title = {'text': "Score"},
        delta = {'reference': threshold, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {'axis': {'range': [0, 1]},
                       'bar': {'color': 'black'},
                        'steps' : [
                         {'range': [0, threshold], 'color': "lightgreen"},
                        {'range': [threshold, 1], 'color':"red"}],
                          'threshold' : {'line': {'color': "yellow", 'width': 4}, 'thickness': 0.75, 'value': threshold}}))
        st.plotly_chart(fig_gauge)

                                              
        return "ok"          
                               
    return "ko"  


       
def infos_client(id_client, df_in):
    with st.container():
        grid_client = st.columns((10))
        
        # Client non répertorié
        valid = is_valid_client(id_client, all_id_client)
        if valid == "ko":
            return "ko"
                    
        else:
            st.markdown("##### Informations client")
            df_data = df_in.copy()
            df_data.drop('Cible', axis=1, inplace=True)
            df_client_int = df_data[df_data['Id client'] == id_client]
            grid_Options = config_aggrid(df_client_int)
            grid_response = AgGrid(
                                    df_client_int, 
                                    gridOptions=grid_Options,
                                    height=80, 
                                    width='100%',
                            )
         
                            
            return "ok"
                    
    return "ko" 
def graphe_infos_client(id_client, df_client_int):
    with st.container():
        grid_client = st.columns((10))
        
        valid = is_valid_client(id_client, all_id_client)
        if valid == "ko":
            return "ko"
                    
        else: 
            data_age = load_age_population(df_client_int)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
            ax.axvline(df_client_int["Age"].values, color="yellow", linestyle='--')
            ax.set(title='Age', xlabel='Age(ans)', ylabel='')
            st.pyplot(fig)
                       
            return "ok"
    return "ko" 
                       


def client_sim(id_client, feature_important_data, select_chart):
    with st.container():
        grid_client_sim = st.columns((10))
        
        # Client non répertorié
        valid = is_valid_client(id_client, all_id_client)
        if valid == "ko":
            return "ko"
                    
        else:
            st.markdown("##### Profils de clients similaires")
            voisin = client_sim_voisins(feature_imp_data)
            if select_chart == "Tableau":
                grid_Options = config_aggrid_2(voisin)
                grid_response = AgGrid(
                                    voisin, 
                                    gridOptions=grid_Options,
                                    height=220, 
                                    width='100%',
                            )
            if select_chart == "Radar Chart":
                radar_chart(voisin)
            
        return "ok"
                    
    return "ko" 

def client_graph_gen(id_client, df_in, var, type_graph):
    with st.container():
        st.markdown("##### Distributions des clients")
        expl, grid_graph_gen_sim = st.columns((1,10))
        
        with expl:
            st.write("")
        with grid_graph_gen_sim:
            if type_graph == 'cible':
            
                if var == 'Age':
                    bar_plot_cible(df_in, 'pl Age client (ans)', 700, 700)
                elif var == '% Annuités/revenus':
                    bar_plot_cible(df_in, 'pl % annuités/revenus', 700, 700)
                else:
                    bar_plot_cible(df_in, var, 600, 600)                
                    
    return "ok" 
    
def client_graph_feat():
    with st.container():
        st.markdown("##### Meilleures variables globales")
        expl, grid_graph_feat = st.columns((1,10))
        
        with expl:
            st.write("")
        with grid_graph_feat:
            image = Image.open('features_importance.png')
            st.image(image)
                    
    return "ok" 

def client_graph_det_feat(data, var):
    with st.container():    
        st.markdown("##### Graphes détaillés sur l\'importance des variables")
        expl, grid_graph_det_feat = st.columns((1,10))
        
        with expl:
            st.write("")
        with grid_graph_det_feat:
            
            if var == 'EXT_SOURCE_2':
                plot_feature_importance(data, 'TARGET', 'EXT_SOURCE_2','Distribution EXT_SOURCE_2', 'TARGET', 'EXT_SOURCE_2')
                image = Image.open('EXT_SOURCE_2.png')
                st.image(image) 

            if var == 'EXT_SOURCE_3':
                plot_feature_importance(data, 'TARGET', 'EXT_SOURCE_3','Distribution EXT_SOURCE_3', 'TARGET', 'EXT_SOURCE_3')
                image = Image.open('EXT_SOURCE_3.png')
                st.image(image)

            if var == 'EXT_SOURCE_1':
                plot_feature_importance(data, 'TARGET', 'EXT_SOURCE_1','Distribution EXT_SOURCE_1', 'TARGET', 'EXT_SOURCE_1')
                image = Image.open('EXT_SOURCE_1.png')
                st.image(image) 

            if var == 'CREDIT_REFUND_TIME':
                plot_feature_importance(data, 'TARGET', 'CREDIT_REFUND_TIME','Distribution CREDIT_REFUND_TIME', 'TARGET', 'CREDIT_REFUND_TIME')
                image = Image.open('CREDIT_REFUND_TIME.png')
                st.image(image)

            if var == 'AGE':
                plot_feature_importance(data, 'CODE_GENDER', 'AGE','Distribution AGE', 'CODE_GENDER', 'AGE')
                image = Image.open('AGE.png')
                st.image(image)          
    return "ok" 
    
class GridOptionsBuilder:

    def __init__(self):
        self.__grid_options = defaultdict(dict)
        self.sideBar = {}

    @staticmethod
    def from_dataframe(dataframe, **default_column_parameters):
       

        type_mapper = {
            "b": ["textColumn"],
            "i": ["numericColumn", "numberColumnFilter"],
            "u": ["numericColumn", "numberColumnFilter"],
            "f": ["numericColumn", "numberColumnFilter"],
            "c": [],
            "m": ['timedeltaFormat'],
            "M": ["dateColumnFilter", "shortDateTimeFormat"],
            "O": [],
            "S": [],
            "U": [],
            "V": [],
        }

        gb = GridOptionsBuilder()
        gb.configure_default_column(**default_column_parameters)

        for col_name, col_type in zip(dataframe.columns, dataframe.dtypes):
            gb.configure_column(field=col_name, type=type_mapper.get(col_type.kind, []))

        return gb

    def configure_default_column(self, min_column_width=5, resizable=True, filterable=True, sorteable=True, editable=False, groupable=False, **other_default_column_properties):
        
        defaultColDef = {
            "minWidth": min_column_width,
            "editable": editable,
            "filter": filterable,
            "resizable": resizable,
            "sortable": sorteable,
        }
        if groupable:
            defaultColDef["enableRowGroup"] = groupable

        if other_default_column_properties:
            defaultColDef = {**defaultColDef, **other_default_column_properties}

        self.__grid_options["defaultColDef"] = defaultColDef

    def configure_auto_height(self, autoHeight=True):
        if autoHeight:
            self.configure_grid_options(domLayout='autoHeight')
        else:
            self.configure_grid_options(domLayout='normal')

    def configure_grid_options(self, **props):
        
        self.__grid_options.update(props)

    def configure_columns(self, column_names=[], **props):
        
        for k in self.__grid_options["columnDefs"]:
            if k in column_names:
                self.__grid_options["columnDefs"][k].update(props)

    def configure_column(self, field, header_name=None, **other_column_properties):
        
        if not self.__grid_options.get("columnDefs", None):
            self.__grid_options["columnDefs"] = defaultdict(dict)

        colDef = {"headerName": header_name if header_name else field, "field": field}

        if other_column_properties:
            colDef = {**colDef, **other_column_properties}

        self.__grid_options["columnDefs"][field].update(colDef)

    def configure_side_bar(self, filters_panel=True, columns_panel=True, defaultToolPanel=""):
        
        filter_panel = {
            "id": "filters",
            "labelDefault": "Filters",
            "labelKey": "filters",
            "iconKey": "filter",
            "toolPanel": "agFiltersToolPanel",
        }

        columns_panel = {
            "id": "columns",
            "labelDefault": "Columns",
            "labelKey": "columns",
            "iconKey": "columns",
            "toolPanel": "agColumnsToolPanel",
        }

        if filters_panel or columns_panel:
            sideBar = {"toolPanels": [], "defaultToolPanel": defaultToolPanel}

            if filters_panel:
                sideBar["toolPanels"].append(filter_panel)
            if columns_panel:
                sideBar["toolPanels"].append(columns_panel)

            self.__grid_options["sideBar"] = sideBar

    def configure_selection(
        self,
        selection_mode="single",
        use_checkbox=False,
        pre_selected_rows=None,
        rowMultiSelectWithClick=False,
        suppressRowDeselection=False,
        suppressRowClickSelection=False,
        groupSelectsChildren=True,
        groupSelectsFiltered=True,
    ):
        
        if selection_mode == "disabled":
            self.__grid_options.pop("rowSelection", None)
            self.__grid_options.pop("rowMultiSelectWithClick", None)
            self.__grid_options.pop("suppressRowDeselection", None)
            self.__grid_options.pop("suppressRowClickSelection", None)
            self.__grid_options.pop("groupSelectsChildren", None)
            self.__grid_options.pop("groupSelectsFiltered", None)
            return

        if use_checkbox:
            suppressRowClickSelection = True
            first_key = next(iter(self.__grid_options["columnDefs"].keys()))
            self.__grid_options["columnDefs"][first_key]["checkboxSelection"] = True
        
        if pre_selected_rows:
            self.__grid_options['preSelectedRows'] = pre_selected_rows

        self.__grid_options["rowSelection"] = selection_mode
        self.__grid_options["rowMultiSelectWithClick"] = rowMultiSelectWithClick
        self.__grid_options["suppressRowDeselection"] = suppressRowDeselection
        self.__grid_options["suppressRowClickSelection"] = suppressRowClickSelection
        self.__grid_options["groupSelectsChildren"] = groupSelectsChildren
        self.__grid_options["groupSelectsFiltered"] = groupSelectsFiltered

    def configure_pagination(self, enabled=True, paginationAutoPageSize=True, paginationPageSize=10):
        
        if not enabled:
            self.__grid_options.pop("pagination", None)
            self.__grid_options.pop("paginationAutoPageSize", None)
            self.__grid_options.pop("paginationPageSize", None)
            return

        self.__grid_options["pagination"] = True
        if paginationAutoPageSize:
            self.__grid_options["paginationAutoPageSize"] = paginationAutoPageSize
        else:
            self.__grid_options["paginationPageSize"] = paginationPageSize

    def build(self):
        
        self.__grid_options["columnDefs"] = list(self.__grid_options["columnDefs"].values())

        return self.__grid_options
        
        
def config_aggrid(df_in):
    # Infer basic colDefs from dataframe types
    gb = GridOptionsBuilder.from_dataframe(df_in)
    
    # Customize gridOptions
    gb.configure_default_column(value=True, enableRowGroup=False, editable=False, filterable=False, sorteable=False)
    
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    
    return gridOptions
    
def config_aggrid_2(df_in):
    # Infer basic colDefs from dataframe types
    gb = GridOptionsBuilder.from_dataframe(df_in)
    
    # Customize gridOptions
    gb.configure_default_column(value=True, enableRowGroup=False, editable=False, filterable=True, sorteable=True)
    gb.configure_pagination(enabled=True, paginationAutoPageSize=True, paginationPageSize=5)
    
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    
    return gridOptions    
   
   
##Clients similaires
def client_sim_voisins(feature_imp_data):
    data_nn = data_train_id_target[data_train_id_target['SK_ID_CURR'] == idclient]
    client_list = std.transform(data_nn[feature_imp_data])  # standardisation
    distance, voisins = nearest_n.kneighbors(client_list)
    voisins = voisins[0]  
    
    # Création d'un dataframe avec les voisins
    clients_similaires = pd.DataFrame()
    for v in range(len(voisins)):
        clients_similaires[v] = data_train_id_target[feature_imp_data].iloc[voisins[v]]   
    voisins_int = pd.DataFrame(index=range(len(clients_similaires.transpose())),
                                       columns=df_int.columns)
                                       
    i = 0
    for id in clients_similaires.transpose()['SK_ID_CURR']:
        voisins_int.iloc[i] = df_int[df_int['Id client'] == id]
        i += 1
           
    return voisins_int
    
def radar_chart(client):
    

    def _invert(x, limits):
        """inverts a value x on a scale from
        limits[0] to limits[1]"""
        return limits[1] - (x - limits[0])

    def _scale_data(data, ranges):
        """scales data[1:] to ranges[0],
        inverts if the scale is reversed"""
        for d, (y1, y2) in zip(data, ranges):
            assert (y1 <= d <= y2) or (y2 <= d <= y1)

        x1, x2 = ranges[0]
        d = data[0]

        if x1 > x2:
            d = _invert(d, (x1, x2))
            x1, x2 = x2, x1

        sdata = [d]

        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            if y1 > y2:
                d = _invert(d, (y1, y2))
                y1, y2 = y2, y1

            sdata.append((d - y1) / (y2 - y1) * (x2 - x1) + x1)

        return sdata

    class ComplexRadar():
        def __init__(self, fig, variables, ranges,
                     n_ordinate_levels=6):
            angles = np.arange(0, 360, (360. / len(variables)))

            axes = [fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True,
                                 label="axes{}".format(i))
                    for i in range(len(variables))]

            axes[0].set_thetagrids(angles, labels=[])

            for ax in axes[1:]:
                ax.patch.set_visible(False)
                ax.grid("off")
                ax.xaxis.set_visible(False)

            for i, ax in enumerate(axes):
                grid = np.linspace(*ranges[i],
                                   num=n_ordinate_levels)                 
                gridlabel = ["{}".format(round(x, 2))
                             for x in grid]
                if ranges[i][0] > ranges[i][1]:
                    grid = grid[::-1]  # hack to invert grid
                    # gridlabels aren't reversed
                gridlabel[0] = ""  # clean up origin
                ax.set_rgrids(grid, labels=gridlabel, angle=angles[i])
                ax.set_ylim(*ranges[i])

            ticks = angles
            ax.set_xticks(np.deg2rad(ticks))  # crée les axes suivant les angles, en radians
            ticklabels = variables
            ax.set_xticklabels(ticklabels, fontsize=10)  # définit les labels

            angles1 = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
            angles1[np.cos(angles1) < 0] = angles1[np.cos(angles1) < 0] + np.pi
            angles1 = np.rad2deg(angles1)
            labels = []
            for label, angle in zip(ax.get_xticklabels(), angles1):
                x, y = label.get_position()
                lab = ax.text(x, y - .5, label.get_text(), transform=label.get_transform(),
                              ha=label.get_ha(), va=label.get_va())
                lab.set_rotation(angle)
                lab.set_fontsize(16)
                lab.set_fontweight('bold')
                labels.append(lab)
            ax.set_xticklabels([])

            # variables for plotting
            self.angle = np.deg2rad(np.r_[angles, angles[0]])
            self.ranges = ranges
            self.ax = axes[0]

        def plot(self, data, *args, **kw):
            sdata = _scale_data(data, self.ranges)
            self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

        def fill(self, data, *args, **kw):
            sdata = _scale_data(data, self.ranges)
            self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw) 

    # data 
    variables = ("% annuités/revenus", 
                "Annuités",
                "Revenus globaux",
                "Age client (ans)",
                "% jours travaillés", 
                "Durée remb crédit (ans)"
                 )
    var_data = ["% annuités/revenus", 
                "Annuités",
                "Revenus globaux", 
                "Age client (ans)",
                "% jours travaillés", 
                "Durée remb crédit (ans)"
                 ]             
    data_ex = client.iloc[0][var_data]
    ranges = [(min(client["% annuités/revenus"]) - 5, max(client["% annuités/revenus"]) + 5),
              (min(client["Annuités"]) - 5000, max(client["Annuités"]) + 5000),
              (min(client["Revenus globaux"]) - 5000, max(client["Revenus globaux"]) + 5000),
              (min(client["Age client (ans)"]) - 5, max(client["Age client (ans)"]) + 5),
              (min(client["% jours travaillés"]) - 5, max(client["% jours travaillés"]) + 5),
              (min(client["Durée remb crédit (ans)"]) - 5, max(client["Durée remb crédit (ans)"]) + 5)
              ]
    
    # plotting
    fig1 = plt.figure(figsize=(7, 7))
    radar = ComplexRadar(fig1, variables, ranges)
    # Affichage des données du client
    radar.plot(data_ex, label='Notre client')
    radar.fill(data_ex, alpha=0.2)
    
    # Affichage de données du client similaires défaillants
    client_defaillant = client[client['Cible'] == 'Client défaillant']
    client_non_defaillant = client[client['Cible'] == 'Client non défaillant']
    
    data = {"% annuités/revenus" : [0.0],
            "Annuités": [0.0],
            "Revenus globaux" : [0.0],
            "Age client (ans)" : [0.0],
            "% jours travaillés" : [0.0],
            "Durée remb crédit (ans)" : [0.0]
           }
    
    client_non_defaillant_mean = pd.DataFrame(data)
    client_defaillant_mean = pd.DataFrame(data)
    
    client_non_defaillant_mean["% annuités/revenus"] = round(client_non_defaillant["% annuités/revenus"].mean(),1)
    client_non_defaillant_mean["Annuités"] = round(client_non_defaillant["Annuités"].mean(),1)
    client_non_defaillant_mean["Revenus globaux"] = round(client_non_defaillant["Revenus globaux"].mean(),1)
    client_non_defaillant_mean["Age client (ans)"] = round(client_non_defaillant["Age client (ans)"].mean(),1)
    client_non_defaillant_mean["% jours travaillés"] = round(client_non_defaillant["% jours travaillés"].mean(),1)
    client_non_defaillant_mean["Durée remb crédit (ans)"] = round(client_non_defaillant["Durée remb crédit (ans)"].mean(),1)
    data_non_client_def = client_non_defaillant_mean.iloc[0][var_data]
    
    client_defaillant_mean["% annuités/revenus"] = round(client_defaillant["% annuités/revenus"].mean(),1)
    client_defaillant_mean["Annuités"] = round(client_defaillant["Annuités"].mean(),1)
    client_defaillant_mean["Revenus globaux"] = round(client_defaillant["Revenus globaux"].mean(),1)
    client_defaillant_mean["Age client (ans)"] = round(client_defaillant["Age client (ans)"].mean(),1)
    client_defaillant_mean["% jours travaillés"] = round(client_defaillant["% jours travaillés"].mean(),1)
    client_defaillant_mean["Durée remb crédit (ans)"] = round(client_defaillant["Durée remb crédit (ans)"].mean(),1)
    data_client_def = client_defaillant_mean.iloc[0][var_data]
     
    radar.plot(data_non_client_def,
               label='Moyenne des clients similaires sans défaut de paiement',
               color='g')
    radar.plot(data_client_def,
               label='Moyenne des clients similaires avec défaut de paiement',
               color='r')           
        
    fig1.legend(bbox_to_anchor=(1.7, 1))

    #st.pyplot(fig1)
    st.pyplot(fig1)
    #st.plotly_chart(fig1)

def bar_plot_cible(df_in, var, width, height):
    df_g = df_in.groupby([var, 'Cible']).size().reset_index()
    df_g['percentage'] = df_in.groupby([var, 'Cible']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
    df_g.columns = [var, 'Cible', 'Nombre', 'Percentage']

    fig = px.bar(df_g, x=var, y='Nombre', color='Cible', text=df_g['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)))
    
    fig.update_layout(
        autosize=True,
        width=width,
        height=height
        )

    st.plotly_chart(fig)
    
def plot_feature_importance(df, col_x, col_y, title, x_label, y_label):
    fig1 = plt.figure(figsize=(5, 5))
    sns.boxplot(x=col_x,
                y=col_y, 
                data=df,
                width=0.5,                
               showmeans=True,
               showfliers=False)
               
    plt.ylabel(y_label, size=10)
    plt.xlabel(x_label, size=10)
    plt.title(title, size=10)
    plt.xticks([0, 1], ['0 : Client non défaillant', '1 : Client défaillant'],
       rotation=0, fontsize=10)
    #plt.tick_params(axis='y', labelsize=3)
    plt.savefig(col_y + '.png')
    #st.pyplot(fig1)
    

    
    
    
st.set_page_config(layout='wide',
                   page_title="Dashboard interactif : décision accord/refus de crédit")
                   
threshold = 0.52




# Affichage des éléments dans la page                   
# Définition des styles pour les éléments de la page
st.markdown(
            '''
            <style>
            .p-style-green {
                font-size:20px;
                font-family:sans-serif;
                color:GREEN;
                vertical-align: text-top;
            }
            
            .p-style-red {
                font-size:20px;
                font-family:sans-serif;
                color:RED;
                vertical-align: top;
            }
            
            .p-style-blue {
                font-size:15px;
                font-family:sans-serif;
                color:BLUE;
                vertical-align: top;
            }
            
            .p-style {
                font-size:15px;
                font-family:sans-serif;
                vertical-align: top;
            }
            
            .p-style-sidebar {
                font-size:17px;
                font-family:sans-serif;
                font-weight: bold;
                vertical-align: top;
                text-decoration: underline;   
            }
            
            .p-style-sidebar-sub {
                font-size:15px;
                font-family:sans-serif;
                vertical-align: top;
                text-decoration: underline;   
            }
            
            </style>
            ''',
            unsafe_allow_html=True

        )
        
# Chargement des données et modèles
lgbm, data, all_id_client, df_int, feature_imp_data, std, nearest_n, data_train_id_target, data_with_target = load_data_model()

st.markdown('# Home Credit Default Risk') 


id_client = st.selectbox("Client ID", all_id_client)



proba_default, prediction = predict_credit_default_client(lgbm, id_client, data, threshold)
if proba_default <= 0.52 :
    decision = "<font color='green'> prêt accordé </font>" 
else:
    decision = "<font color='red'> prêt refusé </font>"

st.write("Decision : ", decision, unsafe_allow_html=True)

st.markdown("<u> Informations client :</u>", unsafe_allow_html=True)
st.write(identite_client(df_int, id_client))






infos_clients = identite_client(df_int, id_client)

        #Age plot
data_age = load_age_population(df_int)
fig, ax = plt.subplots(figsize=(5, 5))
sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
ax.axvline(infos_clients["Age client (ans)"].values, color="yellow", linestyle='--')
ax.set(title='Age Client', xlabel='Age(Year)', ylabel='')
st.pyplot(fig)
    
        

        #Revenue plot
data_income = load_income_population(df_int)
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data_income["Revenus globaux"], edgecolor = 'k', color="blue", bins=10)
ax.axvline(int(infos_clients["Revenus globaux"].values[0]), color="black", linestyle='--')
ax.set(title='Revenus Client', xlabel='Revenue', ylabel='')
st.pyplot(fig)
        
        #Relationship Age / Revenus total plot 
data_sk = data_with_target.reset_index(drop=False)
data_sk.AGE= data_sk['AGE']
fig, ax = plt.subplots(figsize=(10, 10))
fig = px.scatter(data_sk, x='AGE', y="AMT_INCOME_TOTAL", 
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         hover_data=['CNT_CHILDREN', 'AMT_ANNUITY', 'SK_ID_CURR'])

fig.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                          title={'text':"Relationship Age / Revenus Total", 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Revenus Total", title_font=dict(size=18, family='Verdana'))

st.plotly_chart(fig)


shap.initjs()
X = data[data['SK_ID_CURR'] == id_client]
X = X.drop(['SK_ID_CURR'], axis=1)
number = st.slider("Choisissez le nombre de features…", 0, 20, 5)

fig, ax = plt.subplots(figsize=(10, 10))
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
st.pyplot(fig)



idclient = st.sidebar.text_input("Veuillez saisir un ID client :")



if idclient != "":
    idclient = int(idclient)
    
    valid_predict = predict_client(idclient)
    if valid_predict == "ok":
       
        
       # client sim
        lib_client_sim = '<p class="p-style-sidebar">Informations clients similaires:</p>'
        st.sidebar.markdown(lib_client_sim, unsafe_allow_html=True)
            
        clients_with_same_profile = st.sidebar.checkbox('Clients similaires')
        if clients_with_same_profile:
            select_chart = st.sidebar.selectbox("Séléctionnez un type de représentation:", ['Tableau', 'Radar Chart'])
            valid_sim = client_sim(idclient, feature_imp_data, select_chart)
                
            # graphes
            
            selected_option = st.sidebar.multiselect("Sélectionnez une ou plusieurs options:",['Distributions des clients', 
                                                                                      'Les variables importantes', 
                                                                                      'Tous les graphes'])

            if "Tous les graphes" in selected_option:
                selected_option = ['Distributions des clients', 'Les variables importantes']
            
            if "Distributions des clients" in selected_option:    
                lib_graph_gen = '<p class="p-style-sidebar-sub">Distributions des clients:</p>'
                st.sidebar.markdown(lib_graph_gen, unsafe_allow_html=True)        
                gen_var_princ = st.sidebar.selectbox('Sélection de la variable du graphe:',
                                                    ('Genre (H/F)', 'Age', '% Annuités/revenus')) 
                                                    

                if gen_var_princ in ['Genre (H/F)', 'Age', '% Annuités/revenus']:
                    valid_graph_gen = client_graph_gen(idclient, df_int, gen_var_princ, 'cible')
                    

            if "Les variables importantes" in selected_option:
                lib_graph_imp = '<p class="p-style-sidebar-sub">Les variables importantes:</p>'
                st.sidebar.markdown(lib_graph_imp, unsafe_allow_html=True)    
                selected_option_feat = st.sidebar.multiselect('Sélection une ou plusieurs options:',
                                                      ('Les variables importantes', 'Distributions de quelques variables', 'Tous les graphes'))

                if "Tous les graphes" in selected_option_feat:
                    selected_option_feat = ['Les variables importantes', 'Distributions de quelques variables']    
        
                if 'Les variables importantes' in selected_option_feat:
                    valid_graph_feat = client_graph_feat()
                        
                if 'Distributions de quelques variables' in selected_option_feat:
                    gen_var_princ = st.sidebar.selectbox('Sélection de la variable du graphe:',
                                                    ('EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'CREDIT_REFUND_TIME', 'AGE')) 
                    valide_graph_det_feat = client_graph_det_feat(data_with_target, gen_var_princ)
                                


# In[ ]:




