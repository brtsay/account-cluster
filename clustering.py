"""Clustering user accounts"""

import json
import math
import random
import numpy as np
from collections import Counter
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

USERINFO_PATH = 'userinfo_labeled.json'

with open(USERINFO_PATH, 'rb') as jsonfile:
    userinfo = json.load(jsonfile)

# no followers, no followed: follow:followed ratio
# follow few accounts but repost everyday
# no profile picture
# repost without comments all day long
# post very similar things all the time e.g. horoscopes
# activity trend: bursts of activity and then inactivity etc

def make_features(account):
    """Take userinfo and convert to useable features"""
    num_badges = len(account['badges'])
    if 'club' in account.keys():
        club_status = True
        if account['club'] == 'False':
            club_status = False
        elif account['club'] == '微博达人' or account['club'] == 'True': 
            club_status = True
    elif 'club_level' in account.keys():
        club_status = True
    else:
        club_status = False
    if 'credit_num' in account.keys():
        credit = account['credit_num']
    else:
        credit = 0
    level = account['level']
    if 'vip' in account.keys() or 'vip_speed' in account.keys():
        vip = True
    else:
        vip = False
    reg_date = datetime.strptime(account['注册时间'], '%Y-%m-%d')
    account_age = datetime.now() - reg_date
    account_age = account_age.days
    if '关注' in account.keys():
        following_followers = account['关注']/account['粉丝']
    elif account['粉丝'] == 0:
        following_followers = np.inf
    else:
        following_followers = 0
    elapsed_time = datetime.strptime(account['checkTime'], '%Y%m%d-%H%M%S') - reg_date
    if '微博' in account.keys():
        posts_perday = account['微博']/elapsed_time.days
    else:
        posts_perday = 0
    if '标签' in account.keys():
        num_tabs = len(account['标签'])
    else:
        num_tabs = 0
    return [num_badges, club_status, credit, level, vip, account_age, following_followers,
            posts_perday, num_tabs]

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

# using hierarchical clustering
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

# very difficult to see anything, look at random userids
idx = np.random.randint(feature_array.shape[0], size=25)
display_dendrogram(idx)         # look for patterns

# using kmeans
kmeans_n_clusters = 8           # experiment with different k
kmeans = KMeans(n_clusters=kmeans_n_clusters)
kmeans.fit(feature_array)
kmeans_assignment = kmeans.predict(feature_array)

kmeans_cluster_assignment = [[] for x in range(kmeans_n_clusters)]
for i, cluster in enumerate(kmeans_assignment):
    kmeans_cluster_assignment[cluster].append(userids[i])
# look through the resulting object for patterns
