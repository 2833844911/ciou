import numpy as np

def ciou(bboxes1, bboxes2):
    mj1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[1] - bboxes1[3])
    mj2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[1] - bboxes2[3])
    u12 = 0
    zxdx = ((bboxes2[2] - bboxes2[0]) / 2 + bboxes2[0]) - ((bboxes1[2] - bboxes1[0]) / 2 + bboxes1[0])
    zxdy = ((bboxes2[1] - bboxes2[3]) / 2 + bboxes2[3]) - ((bboxes1[1] - bboxes1[3]) / 2 + bboxes1[3])
    zxd = zxdx ** 2 + zxdy ** 2
    maxmj = (max(bboxes1[1], bboxes2[1]) - min(bboxes1[3], bboxes2[3])) ** 2 + (
            max(bboxes1[2], bboxes2[2]) - min(bboxes1[0], bboxes2[0])) ** 2
    xy = [0, 0, 0, 0]
    xy2 = [0, 0, 0, 0]
    if (bboxes2[2] >= bboxes1[0] and bboxes2[0] <= bboxes1[0]):
        xy[0] = 1
    if (bboxes2[2] >= bboxes1[2] and bboxes2[0] <= bboxes1[2]):
        xy[2] = 1
    if (bboxes2[1] >= bboxes1[1] and bboxes2[3] <= bboxes1[1]):
        xy[1] = 1
    if (bboxes2[1] >= bboxes1[3] and bboxes2[3] <= bboxes1[3]):
        xy[3] = 1
    if (bboxes1[2] >= bboxes2[0] and bboxes1[0] <= bboxes2[0]):
        xy2[0] = 1
    if (bboxes1[2] >= bboxes2[2] and bboxes1[0] <= bboxes2[2]):
        xy2[2] = 1
    if (bboxes1[1] >= bboxes2[1] and bboxes1[3] <= bboxes2[1]):
        xy2[1] = 1
    if (bboxes1[1] >= bboxes2[3] and bboxes1[3] <= bboxes2[3]):
        xy2[3] = 1
    # print(xy)
    ks = 0
    if (xy[0] + xy[2]) >= 1 and (xy[1] + xy[3]) >= 1:
        ks = 1
    elif (xy2[0] + xy2[2]) >= 1 and (xy2[1] + xy2[3]) >= 1:
        xy = xy2
        bboxes1, bboxes2 = bboxes2, bboxes1
        ks = 1
    if ks == 1:
        if xy[0] + xy[2] == 2:
            x = abs(bboxes1[2] - bboxes1[0])
        elif xy[0] == 1:
            x = abs(bboxes2[2] - bboxes1[0])
        else:
            x = abs(bboxes1[2] - bboxes2[0])
        if xy[1] + xy[3] == 2:
            y = abs(bboxes1[1] - bboxes1[3])
        elif xy[1] == 1:
            y = abs(bboxes1[1] - bboxes2[3])
        else:
            y = abs(bboxes2[1] - bboxes1[3])

        u12 = x * y
    # print('交：', u12)
    iou = u12 / (mj1 + mj2 - u12)
    v = 4 / (3.141559600830078 ** 2) * (
            np.arctan((bboxes1[2] - bboxes1[0]) / (bboxes1[1] - bboxes1[3])) - np.arctan(
        (bboxes2[2] - bboxes2[0]) / (bboxes2[1] - bboxes2[3]))) ** 2
    arf = v / ((1 - iou) + v)

    lossGiou = 1 - (iou - zxd / maxmj - arf * v)

    # print(lossGiou)
    return lossGiou

box1 = [-423.8458, 1384.2264, 770.5730, -1065.9633]
box2 = [144.5361, 128.4742, 198.9691,  20.0206]
IoU = ciou(np.array(box1), np.array(box2))
print(IoU)


