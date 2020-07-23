"""
Configurations for the entire project
"""

# color setting
COLORS = {
    'face': (255, 255, 0),
    'NBC': (0, 255, 0),
    'NBC exclusive': (255, 0, 0),
    'voice': (0, 255, 255),
    'flick': (255, 0, 255),
    'clevver news': (255, 255, 255),
    'super woman': (255, 255, 255),
    'marvel': (230, 9, 237)
}
# train data
female_path = './data/train_data/female/*.jpg'
male_path = './data/train_data/male/*.jpg'
# xml file for haar cascade
face1 = './xml/haarcascade_frontalface_default.xml'
face2 = './xml/mallick_haarcascade_frontalface_default.xml'
profile1 = './xml/haarcascade_profileface.xml'
profile2 = './xml/mallick_haarcascade_profileface.xml'
train_svm = './xml/haarcascade_frontalface_alt_tree.xml'
# logo template file path
LOGO = {
    'NBC': './pic/Logo/nbc.png',
    'NBC exclusive': './pic/Logo/nbc_exclusive.png',
    'voice': './pic/Logo/the_voice.png',
    'flick': './pic/Logo/flick.png',
    'twitter': './pic/Logo/twitter.jpg',
    'clevver news': './pic/Logo/clevver_news.png',
    'super woman': './pic/Logo/white_super_woman.png',
    'marvel': './pic/Logo/marvel.png',
}
# save labeled image
SAVE_PATH = './save/'
CACHE = './cache/'
CLIP_1 = 'clip_1/'
CLIP_2 = 'clip_2/'
CLIP_3 = 'clip_3/'