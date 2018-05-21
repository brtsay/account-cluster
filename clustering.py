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
    """Take userinfo and convert to useable features"""
    num_badges = len(account['badges'])
    if 'club' in account.keys():
        club_status = True
        if account['club'] == False:
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
    account_age = datetime.now() - reg_date
    account_age = account_age.days
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
    if '标签' in account.keys():
        num_tabs = len(account['标签'])
    else:
        num_tabs = 0
    if '关注' in account.keys():
        following_followers = account['关注']/account['粉丝']
        repost_ratio = account['关注']/posts_perday
    else:
        following_followers = 0
        repost_ratio = 0
    return [num_badges, club_status, credit, level, vip, account_age, following_followers,
            posts_perday, num_tabs, profile_pic, followers, followed, repost_ratio, account['fake']]

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

# using kmeans
kmeans_n_clusters = 8           # experiment with different k
kmeans = KMeans(n_clusters=kmeans_n_clusters)
kmeans.fit(feature_array)
kmeans_assignment = kmeans.predict(feature_array)

kmeans_cluster_assignment = [[] for x in range(kmeans_n_clusters)]
for i, cluster in enumerate(kmeans_assignment):
    kmeans_cluster_assignment[cluster].append(userids[i])
# look through the resulting object for patterns

feature_df = pd.DataFrame.from_records(feature_array)
feature_df.replace([np.inf, -np.inf], np.nan)
feature_df.dropna(inplace=True)

tsne_result = TSNE(learning_rate=100,
                   verbose=2).fit_transform(feature_df)


# the 13th column is the fake indicator column
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=feature_df[13],
            cmap=plt.cm.get_cmap('tab20', 2))
plt.colorbar(ticks=range(2), label='fake')
plt.clim(-0.5, 1.5)
plt.show()
