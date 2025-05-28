# 声音和形状的权重
soundWeight=0.5
shapeWeight=0.5

# 计算字音编码相似性的函数
def computeSoundCodeSimilarity(soundCode1, soundCode2):
    # 特征大小（声音编码的长度）
    featureSize=len(soundCode1)
    # 特征权重
    weights=[0.4,0.4,0.1,0.1]
    multiplier=[]
    # 计算每个特征的相似性
    for i in range(featureSize):
        if soundCode1[i]==soundCode2[i]:
            multiplier.append(1)
        else:
            multiplier.append(0)
    soundSimilarity=0
    # 计算声音编码的相似性
    for i in range(featureSize):
        soundSimilarity += weights[i]*multiplier[i]
    return soundSimilarity
    
# 计算字形编码相似性的函数
def computeShapeCodeSimilarity(shapeCode1, shapeCode2):
    # 特征大小（形状编码的长度）
    featureSize=len(shapeCode1)
    # 特征权重
    weights=[0.15,0.15,0.15,0.15,0.15,0.15,0.1]
    multiplier=[]
    # 计算形状编码的相似性
    for i in range(featureSize):
        if shapeCode1[i]==shapeCode2[i]:
            multiplier.append(1)
        else:
            multiplier.append(0)
    shapeSimilarity=0
    # 计算形状编码的相似性
    for i in range(featureSize):
        shapeSimilarity += weights[i]*multiplier[i]
    return shapeSimilarity

# 计算字符相似性的函数
def computeSSCSimilarity(ssc1, ssc2):
    # 组合字音和字形的相似性，根据权重计算
    shapeSimi=computeShapeCodeSimilarity(ssc1[4:], ssc2[4:])
    soundSimi=computeSoundCodeSimilarity(ssc1[:4], ssc2[:4])
    return max(soundSimi, shapeSimi)
