# 此部分代码源于：https://github.com/zhanlaoban/eda_nlp_for_Chinese
# 用于做easy data augment
import synonyms
import jieba
import random
from random import shuffle
import pandas as pd
import os

random.seed(2022)

DATA_PATH = '../data/'
STOPWORDS_FILENAME = 'cn_stopwords.txt'#停用词文件名

# csv版本
# def load_stopwords_file(filename):
#     print('加载停用词...')
#     stopwords=pd.read_csv(filename, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
#     return set(stopwords['stopword'].values)

# txt版本
def load_stopwords_file_txt(filename):
    print('加载停用词...')
    if not os.path.isfile(filename):
        raise Exception("file does not exist: " + filename)
    content = open(filename, 'rb').read().decode('utf-8')
    stopwords = set()
    for line in content.splitlines():
        stopwords.add(line)
    return stopwords

# 加载停用词表
stop_words = load_stopwords_file_txt(STOPWORDS_FILENAME)
print('停用词表大小：', len(stop_words))

########################################################################
# 同义词替换
# 替换一个语句中的n个单词为其同义词
########################################################################
def synonym_replacement(words, n):
    new_words = words.copy() # 注意改变
    random_word_list = list(set([word for word in words if word not in stop_words])) # 去重
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)# 这不是没变吗
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    return synonyms.nearby(word)[0]


########################################################################
# 随机插入
# 随机在语句中插入n个词
########################################################################
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:  # 三次都相等就摆烂？？
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


########################################################################
# 随机删除
# 以概率p删除语句中的词
########################################################################
def random_deletion(words, p):
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0: # 如果被删的不剩了，随便选一个单词返回
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


########################################################################
# EDA函数
def eda(sentence, segged=False, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    if segged is False:  # 还没有分词
        seg_list = jieba.cut(sentence)  # hide by Jane
        seg_list = " ".join(seg_list)  # hide by Jane
    else:
        seg_list = sentence  # add by Jane
    words = list(seg_list.split())
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1  # 计算可以替换的词语数量
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    # print(words, "\n")

    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))

    # 随机插入ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))

    # 随机交换rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))

    # 随机删除rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(' '.join(a_words))

    # print('*'*20, len(augmented_sentences))
    shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    augmented_sentences.append(seg_list)  # 加上original句子

    return augmented_sentences


if __name__ == '__main__':
    # 测试用例
    """
    sentence = "我们就像蒲公英，我也祈祷着能和你飞去同一片土地"
    augmented_sentences = eda(sentence)
    print(augmented_sentences)
    """
    seg_list = "我们 就 像 蒲公英 ， 我 也 祈祷 着 能 和 你 飞去 同 一片 土地"
    augmented_sentences = eda(seg_list, segged=True, alpha_sr=0.05, alpha_ri=0.05, alpha_rs=0.05, p_rd=0.05, num_aug=9)
    print(augmented_sentences)

