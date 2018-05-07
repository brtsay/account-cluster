"""Clustering user accounts"""

import json
import math
import random
import numpy as np
from collections import Counter
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

plt.interactive(False)
plt.show(block=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
print('x' in np.arange(5))

USERINFO_PATH = 'userinfo_labeled.json'

with open(USERINFO_PATH, 'rb') as jsonfile:
    userinfo = json.load(jsonfile)
    
# looking at values in single "column"
profile_pics = [account['profile_pic'] for account in userinfo if 'profile_pic' in account.keys()]
# profile_pics = []
# for account in userinfo:
#     if 'profile_pic' in account.keys():
#         profile_pics.append(account['profile_pic'])
# see first 5 entries
profile_pic_counts = Counter(profile_pics)
high_counts = [url for url, count in profile_pic_counts.items() if count > 1]
print(high_counts)
    
# no followers, no followed: follow:followed ratio
# follow few accounts but repost everyday
# no profile picture
# repost without comments all day long
# post very similar things all the time e.g. horoscopes
# activity trend: bursts of activity and then inactivity etc

def make_features(account):
    if 'profile_pic' in account.keys():
        profile_pic = True
        if account ['profile_pic'] == '//ww1.sinaimg.cn/default/images/default_avatar_male_uploading_180.gif':
            profile_pic = False
        elif account['profile_pic'] == '//ww1.sinaimg.cn/default/images/default_avatar_female_uploading_180.gif':
            profile_pic = False
        else:
            profile_pic = True
    else:
        profile_pic = False
    reg_date = datetime.strptime(account['注册时间'], '%Y-%m-%d')
    elapsed_time = datetime.strptime(account['checkTime'], '%Y%m%d-%H%M%S') - reg_date
    if '关注' in account.keys():
        followers = True
        if account['关注'] == 0:
            followers = False
        else:
                followers = True
    else:
        followers = False
    if '粉丝' in account.keys():
        followed = True
        if account['粉丝'] == 0:
            followed = False
        else:
            followed = True
    else:
        followed = False
    if '微博' in account.keys():
        posts_perday = account['微博'] / elapsed_time.days
    else:
        posts_perday = 0
    if '关注' in account.keys():
        repost_ratio = account['关注']/posts_perday
    else:
        repost_ratio = 0
    return [profile_pic, followers, followed, repost_ratio]

feature_list = []
userids = []
for account in userinfo:
    try:
        feature_list.append(make_features(account))
        userids.append(account['userid'])
    except (KeyError, ZeroDivisionError) as err:
        print(err)
        print(account)
feature_array = np.array(feature_list)

print(feature_list)

def display_dendrogram(idx='all'):
    """Run hierarchical clustering with Ward linkage and show clustogram"""
    if idx == 'all':
        to_cluster = feature_array
        the_labels = userids
    else:
        to_cluster = feature_array[idx]
        the_labels = np.array(userids)[idx]
    hier_clust = linkage(to_cluster, method='ward')
    dendrogram(hier_clust, labels=the_labels, orientation='left')

idx = np.random.randint(feature_array.shape[0], size=25)
display_dendrogram(idx)
plt.show()
