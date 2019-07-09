# -*- coding: utf-8 -*-

from __future__ import print_function
import pandas as pd
import numpy as np
import math
import copy
import random
import glob
import os

unit_size = 5
feature_dim = 2048 + 1024


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def getBatchList(numWindow, batch_size, shuffle=True):
    ## notice that there are some video appear twice in last two batch ##
    window_list = range(numWindow)
    batch_start_list = [i * batch_size for i in range(len(window_list) / batch_size)]
    batch_start_list.append(len(window_list) - batch_size)
    if shuffle == True:
        random.shuffle(window_list)
    batch_window_list = []
    for bstart in batch_start_list:
        batch_window_list.append(window_list[bstart:(bstart + batch_size)])
    return batch_window_list


def getVideoFeature(videoname, subset):
    appearance_path = '/home/litao/THUMOS14_ANET_feature/{}_appearance/'.format(subset)
    denseflow_path = '/home/litao/THUMOS14_ANET_feature/{}_denseflow/'.format(subset)
    rgb_feature = np.load(appearance_path + videoname + '.npy')
    flow_feature = np.load(denseflow_path + videoname + '.npy')

    return rgb_feature, flow_feature


def getBatchData(window_list, data_dict):
    batch_info = []
    batch_anchor_feature = []
    for idx in window_list:
        batch_info.append(data_dict["info"][idx])
        batch_anchor_feature.append(data_dict["feature"][idx])
    batch_anchor_feature = np.array(batch_anchor_feature)

    return batch_anchor_feature, batch_info


def getFullData(dataSet):
    ii = 0
    # dataSet="Test"
    annoDf = pd.read_csv("./data/thumos_14_annotations/" + dataSet + "_Annotation.csv")
    videoNameList = list(set(annoDf.video.values[:]))
    list_data = []
    list_info = []
    for videoName in videoNameList:
        print('complete {}/{}.'.format(ii, len(videoNameList)), end='\r')
        ii += 1
        rgb_feature, flow_feature = getVideoFeature(videoName, dataSet.lower())
        numSnippet = min(rgb_feature.shape[0], flow_feature.shape[0])
        frameList = [1 + unit_size * i for i in range(numSnippet)]
        df_data = np.concatenate((rgb_feature, flow_feature), axis=1)
        df_snippet = frameList
        window_size = 128
        stride = window_size / 2
        n_window = (numSnippet + stride - window_size) / stride
        windows_start = [i * stride for i in range(n_window)]
        if numSnippet < window_size:
            windows_start = [0]
            tmp_data = np.zeros((window_size - numSnippet, feature_dim))
            df_data = np.concatenate((df_data, tmp_data), axis=0)
            df_snippet.extend([df_snippet[-1] + unit_size * (i + 1) for i in range(window_size - numSnippet)])
        else:
            windows_start.append(numSnippet - window_size)

        snippet_xmin = df_snippet
        snippet_xmax = df_snippet[1:]
        snippet_xmax.append(df_snippet[-1] + unit_size)
        for start in windows_start:
            tmp_data = df_data[start:start + window_size, :]
            tmp_anchor_xmins = snippet_xmin[start:start + window_size]
            list_data.append(tmp_data)
            list_info.append([tmp_anchor_xmins[0], videoName])
    dataDict = {"info": list_info,
                "feature": list_data}
    return dataDict


def getVideoData(videoName, subset):
    list_data = []
    list_snippets = []
    rgb_feature, flow_feature = getVideoFeature(videoName, subset)
    numSnippet = min(len(rgb_feature), len(flow_feature))
    frameList = [1 + unit_size * i for i in range(numSnippet)]
    df_data = np.concatenate((rgb_feature, flow_feature), axis=1)
    df_snippet = frameList
    window_size = 128
    stride = window_size / 2
    n_window = (numSnippet + stride - window_size) / stride
    windows_start = [i * stride for i in range(n_window)]

    if numSnippet < window_size:
        windows_start = [0]
        tmp_data = np.zeros((window_size - numSnippet, feature_dim))
        df_data = np.concatenate((df_data, tmp_data), axis=0)
        df_snippet.extend([df_snippet[-1] + unit_size * (i + 1) for i in range(window_size - numSnippet)])
    else:
        windows_start.append(numSnippet - window_size)

    for start in windows_start:
        tmp_data = df_data[start:start + window_size, :]
        tmp_snippets = np.array(df_snippet[start:start + window_size])
        list_data.append(tmp_data)
        list_snippets.append(tmp_snippets)

    list_snippets = np.array(list_snippets)
    list_data = np.array(list_data)
    return list_snippets, list_data, df_snippet
