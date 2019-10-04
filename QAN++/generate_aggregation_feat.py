import numpy as np
import pdb
import sys
video_list = 'video_filelist.txt'
img_list = 'list.txt'

video_list = open(video_list).readlines()
video_list = [x.split(' ')[0] for x in video_list]

img_list = open(img_list).readlines()

feat_name = sys.argv[1]
feat_dim = int(sys.argv[2])
quality_name = sys.argv[3]
protocol = 'SQA'

feat = np.fromfile(feat_name,dtype=np.float32)
feat = feat.reshape(-1,feat_dim)

out_feat_name = feat_name+'_'+protocol+'_agre'

print('generate_feat_dict~~~~~~')

video_dict = dict()
for i in range(len(img_list)):
    if i%100000==0:
        print('process dict:{}/{}'.format(i,len(img_list)))
    video_name = img_list[i].split('/')[1]
    if video_name in video_dict.keys():
        video_dict[video_name].append(i)
    else:
        video_dict[video_name]=[]
        video_dict[video_name].append(i)
print('start aggregate feature~~~')

video_feat = []
if protocol=='SQA':
    quality = np.fromfile(quality_name,dtype=np.float32)
    for idx,key in enumerate(video_list):
        if idx%10000==0:
            print('process feature with {}:{}/{}'.format(protocol,idx,len(video_list)))
        img_idxes = np.array(video_dict[key])
        sub_feats = feat[img_idxes,:]
        sub_quality = quality[img_idxes].reshape(-1,1)
        if len(sub_quality)>2:
            min_value = sub_quality.min()
            max_value = sub_quality.max()
            K = 1/(max_value-min_value)
            B = 1-K*max_value
            sub_quality = K*sub_quality+B

        sub_feats = sub_feats*sub_quality/sub_quality.sum()
        video_feat.append(sub_feats.sum(axis=0))

video_feat = np.array(video_feat)

video_feat.tofile(out_feat_name)
