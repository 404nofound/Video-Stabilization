import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os
from PIL import Image
import PIL
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
# get param
name='brisk'
parser = argparse.ArgumentParser(description='')
parser.add_argument('-v', type=str, default=r'C:\\Users\\Eddy\\Desktop\\test.mp4')  # 指定输入视频路径位置（参数必选）
parser.add_argument('-o', type=str, default=r'C:\\Users\\Eddy\\Desktop\\output\\'+name+'_out.mp4')  # 指定输出视频路径位置（参数必选）
parser.add_argument('-n', type=int, default=-1)  # 指定处理的帧数（参数可选）, 不设置使用视频实际帧
parser.add_argument('-model', type=str, default=name)  # 指定处理的帧数（参数可选）, 不设置使用视频实际帧

# eg: python3 stable.py -v=video/01.mp4 -o=video/01_stable.mp4 -n=100 -p=6

args = parser.parse_args()
#args = parser.parse_known_args()[0]

input_path = args.v
output_path = args.o
number = args.n
model = args.model

if model not in ['sift','surf','brisk','freak','orb','akaze']:
    print('Wrong model')
    sys.exit()

#default
#index_params = dict(algorithm=6, table_number=6, key_size=12)

if model=='sift' or model=='surf':
    index_params = dict(algorithm=0, trees=5)
elif model=='orb':
    index_params = dict(algorithm=6, table_number=6, key_size=12)
else:
    index_params = dict(algorithm=6, table_number=6, key_size=12)

def keypointToPoint(keypoint):
    point = np.zeros(len(keypoint) * 2, np.float32)
    for i in range(len(keypoint)):
        point[i * 2] = keypoint[i].pt[0]
        point[i * 2 + 1] = keypoint[i].pt[1]
    point = point.reshape(-1, 2)
    return point

def drawMatchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0,0,255))

    img11=Image.fromarray(vis)
    img11.save(r'C:\\Users\\Eddy\\Desktop\\match.png')
    #cv2.namedWindow("match",cv2.WINDOW_NORMAL)
    #cv2.imshow("match", vis)

class Stable:
    # 处理视频文件路径
    __input_path = None

    __output_path = None

    __number = number

    # surf 特征提取
    __surf = {
        # surf算法
        'surf': None,
        # 提取的特征点
        'kp': None,
        # 描述符
        'des': None,
        # 过滤后的特征模板
        'template_kp': None
    }

    # capture
    __capture = {
        # 捕捉器
        'cap': None,
        # 视频大小
        'size': None,
        # 视频总帧
        'frame_count': None,
        # 视频帧率
        'fps': None,
        # 视频
        'video': None,
    }

    # 配置
    __config = {
        # 要保留的最佳特征的数量
        'key_point_count': 5000,
        # Flann特征匹配
        #SIFT and SURF
        #'index_params': dict(algorithm=0, trees=5),
        #ORB
        #FLANN_INDEX_LSH=6
        #'index_params': dict(algorithm=6, table_number=6, key_size=12),
        'index_params': index_params,
        'search_params': dict(checks=50),
        'ratio': 0.5,
    }

    # 特征提取列表
    __surf_list = []

    def __init__(self):
        pass

    # 初始化capture
    def __init_capture(self):
        self.__capture['cap'] = cv2.VideoCapture(self.__video_path)
        self.__capture['size'] = (int(self.__capture['cap'].get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(self.__capture['cap'].get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.__capture['fps'] = self.__capture['cap'].get(cv2.CAP_PROP_FPS)

        self.__capture['video'] = cv2.VideoWriter(self.__output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                  self.__capture['fps'], self.__capture['size'])

        self.__capture['frame_count'] = int(self.__capture['cap'].get(cv2.CAP_PROP_FRAME_COUNT))

        if number == -1:
            self.__number = self.__capture['frame_count']
        else:
            self.__number = min(self.__number, self.__capture['frame_count'])

    # 初始化surf
    def __init_surf(self):

        self.__capture['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
        state, first_frame = self.__capture['cap'].read()

        self.__capture['cap'].set(cv2.CAP_PROP_POS_FRAMES, self.__capture['frame_count'] - 1)
        state, last_frame = self.__capture['cap'].read()

        # mask
        mask=np.zeros([first_frame.shape[0],first_frame.shape[1]],dtype=np.uint8)
        mask[143:1500,508:1700]=255
        first_frame=cv2.add(first_frame,np.zeros(np.shape(first_frame),dtype=np.uint8),mask=mask)
        last_frame=cv2.add(last_frame,np.zeros(np.shape(last_frame),dtype=np.uint8),mask=mask)

        # default model ORB


        if model=='sift':
            self.__surf['surf'] = cv2.xfeatures2d.SIFT_create(self.__config['key_point_count'])
        elif model=='surf':
            self.__surf['surf'] = cv2.xfeatures2d.SURF_create(self.__config['key_point_count'])
        elif model=='orb':
            self.__surf['surf'] = cv2.ORB_create(self.__config['key_point_count'])
        elif model=='brisk':
            self.__surf['surf'] = cv2.BRISK_create()
        elif model=='freak':
            self.__surf['surf'] = cv2.Freak_create()
        elif model=='akaze':
            self.__surf['surf'] = cv2.AKAZE_create()

        self.__surf['kp'], self.__surf['des'] = self.__surf['surf'].detectAndCompute(first_frame, None)
        kp, des = self.__surf['surf'].detectAndCompute(last_frame, None)

        # for p in keypointToPoint(self.__surf['kp']):
        #     cv2.circle(first_frame, (p[0],p[1]), 3, (0, 0, 255), -1)
        #
        # im_f=Image.fromarray(first_frame)
        # im_f.save('C:\\Users\\Eddy\\Desktop\\first5.png')
        #
        # for p in keypointToPoint(kp):
        #     cv2.circle(last_frame, (p[0],p[1]), 3, (0, 0, 255), -1)
        #
        # im_f=Image.fromarray(last_frame)
        # im_f.save('C:\\Users\\Eddy\\Desktop\\last5.png')

        #img1=cv2.drawKeypoints(first_frame,self.__surf['kp'],None,(255,0,0),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        img1=cv2.drawKeypoints(first_frame,self.__surf['kp'],None,(255,0,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #img1=cv2.drawKeypoints(first_frame,self.__surf['kp'],None,(255,0,0),cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        #img1=cv2.drawKeypoints(first_frame,self.__surf['kp'],None,(255,0,0),cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINT)
        img1=Image.fromarray(img1)
        img1.save(r'C:\\Users\\Eddy\\Desktop\\output\\'+model+'_first_frame.png')

        #img2=cv2.drawKeypoints(last_frame,kp,None,(255,0,0),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        img2=cv2.drawKeypoints(last_frame,kp,None,(255,0,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #img2=cv2.drawKeypoints(last_frame,kp,None,(255,0,0),cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        #img2=cv2.drawKeypoints(last_frame,kp,None,(255,0,0),cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINT)
        img2=Image.fromarray(img2)
        img2.save(r'C:\\Users\\Eddy\\Desktop\\output\\'+model+'_last_frame.png')


        # 快速临近匹配
        flann = cv2.FlannBasedMatcher(self.__config['index_params'], self.__config['search_params'])
        matches = flann.knnMatch(self.__surf['des'], des, k=2)

        good_match = []
        for m, n in matches:
            if m.distance < self.__config['ratio'] * n.distance:
                good_match.append(m)

        #drawMatchesKnn_cv2(first_frame,self.__surf['kp'],last_frame,kp,good_match)
        img_out=cv2.drawMatches(first_frame,self.__surf['kp'],last_frame,kp,good_match[:50],last_frame,flags=2)
        img_out=Image.fromarray(img_out)
        img_out.save(r'C:\\Users\\Eddy\\Desktop\\output\\'+model+'_match.png')

        self.__surf['template_kp'] = []
        for f in good_match:
            self.__surf['template_kp'].append(self.__surf['kp'][f.queryIdx])

    # 释放
    def __release(self):
        self.__capture['video'].release()
        self.__capture['cap'].release()

    # 处理
    def __process(self):

        current_frame = 1

        self.__capture['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)

        process_bar = tqdm(self.__number, position=current_frame)

        while current_frame <= self.__number:
            # 抽帧
            success, frame = self.__capture['cap'].read()

            if not success: return

            # 计算
            frame = self.detect_compute(frame)

            # 写帧
            self.__capture['video'].write(frame)

            current_frame += 1

            process_bar.update(1)

    # 视频稳像
    def stable(self, input_path, output_path, number):
        self.__video_path = input_path
        self.__output_path = output_path
        self.__number = number

        self.__init_capture()
        self.__init_surf()
        self.__process()
        self.__release()

    # 特征点提取
    def detect_compute(self, frame):

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算特征点
        kp, des = self.__surf['surf'].detectAndCompute(frame_gray, None)

        # 快速临近匹配
        flann = cv2.FlannBasedMatcher(self.__config['index_params'], self.__config['search_params'])
        matches = flann.knnMatch(self.__surf['des'], des, k=2)

        # 计算单应性矩阵
        good_match = []
        for m, n in matches:
            if m.distance < self.__config['ratio'] * n.distance:
                good_match.append(m)

        # 特征模版过滤
        p1, p2 = [], []
        for f in good_match:
            if self.__surf['kp'][f.queryIdx] in self.__surf['template_kp']:
                p1.append(self.__surf['kp'][f.queryIdx].pt)
                p2.append(kp[f.trainIdx].pt)

        # 单应性矩阵
        H, _ = cv2.findHomography(np.float32(p2), np.float32(p1), cv2.RHO)

        # 透视变换
        output_frame = cv2.warpPerspective(frame, H, self.__capture['size'], borderMode=cv2.BORDER_REPLICATE)

        return output_frame


if __name__ == '__main__':

    if not os.path.exists(input_path):
        print(f'[ERROR] File "{input_path}" not found')
        exit(0)
    else:
        print(f'[INFO] Video "{input_path}" stable begin')

    s = Stable()
    s.stable(input_path, output_path, number)

    print('[INFO] Done.')
    exit(0)
