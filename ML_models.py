# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 09:45:35 2021

@author: Nicos Evmides & Brian
"""
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import yaml
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
#from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path

curr_dir = Path(__file__).parent.absolute()
hyperparameters_cnfg_dir = rf"{curr_dir}/config files/optimal hyperparameters"



def get_dtr_model(splitter_inp, max_depth_inp, min_samples_split_inp, min_samples_leaf_inp, min_weight_fraction_leaf_inp, max_features_inp, ccp_alpha_inp):
    mach_learn_mdl = DecisionTreeRegressor(
	splitter = splitter_inp,
	max_depth = max_depth_inp,
	min_samples_split = min_samples_split_inp,
	min_samples_leaf = min_samples_leaf_inp,
	min_weight_fraction_leaf = min_weight_fraction_leaf_inp,
	max_features = max_features_inp,
	ccp_alpha = ccp_alpha_inp
	)
    return mach_learn_mdl


def get_dtrp1_model(splitter_inp, max_depth_inp, min_samples_split_inp, min_samples_leaf_inp, min_weight_fraction_leaf_inp, max_features_inp, ccp_alpha_inp):
    mach_learn_mdl = DecisionTreeRegressor(
	splitter = splitter_inp,
	max_depth = max_depth_inp,
	min_samples_split = min_samples_split_inp,
	min_samples_leaf = min_samples_leaf_inp,
	min_weight_fraction_leaf = min_weight_fraction_leaf_inp,
	max_features = max_features_inp,
	ccp_alpha = ccp_alpha_inp
	)
    return mach_learn_mdl

def get_knn_model(n_neighbors_inp, weights_inp, algorithm_inp, leaf_size_inp, p_inp):
    mach_learn_mdl = KNeighborsRegressor(
    	n_neighbors=n_neighbors_inp,
	weights=weights_inp,
 	algorithm=algorithm_inp,
	leaf_size=leaf_size_inp,
	p=p_inp
	)
    return mach_learn_mdl

def get_knnp1_model(n_neighbors_inp, weights_inp, algorithm_inp, leaf_size_inp, p_inp):
    mach_learn_mdl = KNeighborsRegressor(
    	n_neighbors=n_neighbors_inp,
	weights=weights_inp,
 	algorithm=algorithm_inp,
	leaf_size=leaf_size_inp,
	p=p_inp
	)
    return mach_learn_mdl

def get_gaussian_model(var_smoothing_inp):
    mach_learn_mdl = GaussianNB(
        var_smoothing=var_smoothing_inp)
    return mach_learn_mdl


def get_nn_model(hidden_layer_sizes1_inp, hidden_layer_sizes2_inp, hidden_layer_sizes3_inp, activation_inp, alpha_inp, learning_rate_inp, solver_inp, max_iter_inp):
    mach_learn_mdl = MLPRegressor(
	hidden_layer_sizes=(hidden_layer_sizes1_inp,hidden_layer_sizes2_inp,hidden_layer_sizes3_inp),
	activation=activation_inp,
	alpha=alpha_inp,
	learning_rate=learning_rate_inp,
	solver=solver_inp,
	max_iter=max_iter_inp)
    return mach_learn_mdl

def get_nnp1_model(hidden_layer_sizes1_inp, hidden_layer_sizes2_inp, activation_inp, alpha_inp, learning_rate_inp, learning_rate_init_inp, solver_inp, max_iter_inp):
    mach_learn_mdl = MLPRegressor(
	hidden_layer_sizes=(hidden_layer_sizes1_inp,hidden_layer_sizes2_inp),
	activation=activation_inp,
	alpha=alpha_inp,
	learning_rate=learning_rate_inp,
        learning_rate_init=learning_rate_init_inp,
	solver=solver_inp,
	max_iter=max_iter_inp)
    return mach_learn_mdl

def get_nnp3_model(hidden_layer_sizes1_inp, max_iter_inp):
    mach_learn_mdl = MLPRegressor(
	hidden_layer_sizes=hidden_layer_sizes1_inp,
        max_iter=max_iter_inp
	)
    return mach_learn_mdl


def get_svm_model(deg, gam, ker):
    mach_learn_mdl = SVR(degree=deg,
                         gamma=gam,
                         kernel=ker)
    return mach_learn_mdl


def get_rf_model(trees, max_feat, max_dep, bootstrap, min_smpl_splt,
                 min_smpl_lf):
    mach_learn_mdl = RandomForestRegressor(n_estimators=trees,
                                           max_features=max_feat,
                                           max_depth=max_dep,
                                           bootstrap=bootstrap,
                                           min_samples_split=min_smpl_splt,
                                           min_samples_leaf=min_smpl_lf)
    return mach_learn_mdl


def get_gbr_model(trees, learn_rt, max_dep, min_smpl_splt, max_feat, subsmpl):
    mach_learn_mdl = GradientBoostingRegressor(n_estimators=trees,
                                               learning_rate=learn_rt,
                                               max_depth=max_dep,
                                               min_samples_split=min_smpl_splt,
                                               max_features=max_feat,
                                               subsample=subsmpl)
    return mach_learn_mdl

def get_gbrp2_model(learn_rt, max_dep):
    mach_learn_mdl = GradientBoostingRegressor(learning_rate=learn_rt,
                                               max_depth=max_dep,
                                               )
    return mach_learn_mdl


def get_xgbr_model(trees, learn_rt, max_dep, min_chld_wght, colsample,
                   subsmpl):
    mach_learn_mdl = XGBRegressor(n_estimators=trees,
                                  learning_rate=learn_rt,
                                  max_depth=max_dep,
                                  min_child_weight=min_chld_wght,
                                  colsample_bytree=colsample,
                                  subsample=subsmpl)
    return mach_learn_mdl


def read_mdl_optimal_hyper_params(model_shortname):
    hyper_prms_fname = f'{hyperparameters_cnfg_dir}/optimized_{model_shortname}_hyperparms.yml'
    with open(hyper_prms_fname) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d


def get_tuned_model(model_shortname):
    optimal_hyper_params = read_mdl_optimal_hyper_params(model_shortname)
    mach_learn_mdl = None
    if model_shortname == 'dtr':
        mach_learn_mdl = get_dtr_model(**optimal_hyper_params)
    if model_shortname == 'dtrp1':
        mach_learn_mdl = get_dtrp1_model(**optimal_hyper_params)
    if model_shortname == 'knn':
        mach_learn_mdl = get_knn_model(**optimal_hyper_params)
    if model_shortname == 'knnp1':
        mach_learn_mdl = get_knnp1_model(**optimal_hyper_params)
    if model_shortname == 'gaussian':
        mach_learn_mdl = get_gaussian_model(**optimal_hyper_params)
    if model_shortname == 'nn':
        mach_learn_mdl = get_nn_model(**optimal_hyper_params)
    if model_shortname == 'nnp1':
        mach_learn_mdl = get_nnp1_model(**optimal_hyper_params)
    if model_shortname == 'nnp3':
        mach_learn_mdl = get_nnp3_model(**optimal_hyper_params)
    if model_shortname == 'svm':
        mach_learn_mdl = get_svm_model(**optimal_hyper_params)
    if model_shortname == 'gbr':
        mach_learn_mdl = get_gbr_model(**optimal_hyper_params)
    if model_shortname == 'gbrp2':
        mach_learn_mdl = get_gbrp2_model(**optimal_hyper_params)
    if model_shortname == 'xgbr':
        mach_learn_mdl = get_xgbr_model(**optimal_hyper_params)
    if model_shortname == 'rf':
        mach_learn_mdl = get_rf_model(**optimal_hyper_params)
    return mach_learn_mdl
