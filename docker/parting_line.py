import cv2
import numpy as np


# 曲线二项式拟合及按曲率采样,返回MES及采样点索引
def binomial_fit(x, y, sample=20, ploy_threshold=5):
    ploys = []
    for i in range(1, ploy_threshold):
        z = np.polyfit(x, y, i)  # 曲线拟合
        # p = np.poly1d(z)  # 数学函数
        # dp = p.deriv()  # 一阶导数
        # ddp = dp.deriv()  # 二阶导数
        # 计算多项式
        x_fit = np.polyval(z, x)
        # 原值与拟合值的均方差
        mse = np.linalg.norm(x_fit - y, ord=2) ** 2 / x_fit.shape[0]
        ploys.append({'z': z, 'mse': mse})
    print('ploys:', ploys)
    min_ploy = ploys[0]
    for ploy in ploys:
        if ploy['mse'] < min_ploy['mse']:
            min_ploy = ploy

    p = np.poly1d(min_ploy['z'])  # 数学函数
    dp = p.deriv()  # 一阶导数
    ddp = dp.deriv()  # 二阶导数
    # 求曲率
    curvatures = abs(ddp(x)) / ((1 + dp(x) ** 2) ** 1.5)  # 曲率
    # 曲率积分图
    sum_curvatures = [curvatures[0]]
    for curvature in curvatures[1:]:
        sum_curvatures.append(sum_curvatures[-1] + curvature)
    # 曲率积分等差分割值
    seg_values = np.linspace(sum_curvatures[0], sum_curvatures[-1], num=sample, endpoint=True)
    # 采样点索引值
    sample_indexes = [0]
    for i in range(1, seg_values.shape[0]):
        sample_indexes.append(np.argwhere(
            (sum_curvatures > seg_values[i - 1]) & (sum_curvatures <= seg_values[i])).reshape(-1)[-1])
    print('采样点索引值:', sample_indexes)

    fit_points = np.hstack((x[sample_indexes].reshape(-1, 1), x_fit[sample_indexes].reshape(-1, 1)))

    return min_ploy['mse'], fit_points


# 鱼头及鱼腹切割线
def head_belly_line(mask_head, mask_tail, mask_body):
    # 寻找轮廓, contours 从最上方开始,逆时针旋转
    contours_head, _ = cv2.findContours(mask_head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_tail, _ = cv2.findContours(mask_tail, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_body, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_head2 = np.array(contours_head).reshape(-1, 2)
    contours_tail2 = np.array(contours_tail).reshape(-1, 2)
    contours_body2 = np.array(contours_body).reshape(-1, 2)

    # 多边形拟合, epsilon_***参数待定
    head_size = contours_head2.max(axis=0) - contours_head2.min(axis=0)  # = 20
    tail_size = contours_tail2.max(axis=0) - contours_tail2.min(axis=0)  # = 20
    body_size = contours_body2.max(axis=0) - contours_body2.min(axis=0)  # = 20

    epsilon_head = head_size.min() // 3
    epsilon_tail = tail_size.min() // 3
    epsilon_body = body_size.min() // 3
    approx_head = cv2.approxPolyDP(contours_head[0], epsilon_head, True)
    approx_tail = cv2.approxPolyDP(contours_tail[0], epsilon_tail, True)
    approx_body = cv2.approxPolyDP(contours_body[0], epsilon_body, True)

    approx_head2 = np.array(approx_head).reshape(-1, 2)
    approx_tail2 = np.array(approx_tail).reshape(-1, 2)
    approx_body2 = np.array(approx_body).reshape(-1, 2)

    # 多边形求重心, 然后求各顶点相对重心的角度
    approx_head_centre = np.mean(approx_head2, axis=0)
    approx_tail_centre = np.mean(approx_tail2, axis=0)
    approx_head_mag, approx_head_ang = cv2.cartToPolar(approx_head2[..., 0] - approx_head_centre[0],
                                                       approx_head2[..., 1] - approx_head_centre[1])
    approx_tail_mag, approx_tail_ang = cv2.cartToPolar(approx_tail2[..., 0] - approx_tail_centre[0],
                                                       approx_tail2[..., 1] - approx_tail_centre[1])
    # 弧度转角度
    approx_head_deg = np.rad2deg(approx_head_ang.reshape(-1))
    approx_tail_deg = np.rad2deg(approx_tail_ang.reshape(-1))

    if approx_head_centre[0] < approx_tail_centre[0]:
        print('头部在左边: {}, {}'.format(
            approx_head_centre.astype(np.int32), approx_tail_centre.astype(np.int32)))
        # 寻找 approx_head 右上角的点
        head_ang_order = np.argsort(approx_head_deg)[::-1]
        head_right_up = approx_head2[head_ang_order[0]]
        # 寻找 approx_tail 左上角的点
        tail_rotated_deg = approx_tail_deg + 90  # 坐标系逆时针旋转90度,左上角在第四象限
        approx_tail_deg = np.where(tail_rotated_deg < 360, tail_rotated_deg, tail_rotated_deg - 360)
        tail_ang_order = np.argsort(approx_tail_deg)[::-1]
        tail_left_up = approx_tail2[tail_ang_order[0]]

        # 求鱼腹坐标点集
        belly_index = np.argwhere((contours_body2[:, 0] >= head_right_up[0]) &
                                  (contours_body2[:, 0] <= tail_left_up[0]) &
                                  (contours_body2[:, 1] <= tail_left_up[1]))

        belly_points = contours_body2[belly_index].reshape(-1, 2)
        # 鱼腹坐标点集按x排序
        belly_order = np.argsort(belly_points[:, 0])
        belly_points_ordered = belly_points[belly_order]

        # 求鱼头切割点集
        head_separator_start = np.argmax(contours_head2[:, 1])
        head_points = contours_head2[head_separator_start:]
        head_points_ordered = head_points[::-1]  # 倒序

        return head_points_ordered, belly_points_ordered

    else:
        print('头部在右边: {}, {}'.format(
            approx_head_centre.astype(np.int32), approx_tail_centre.astype(np.int32)))
        # 寻找 approx_head 左上角的点
        head_rotated_deg = approx_head_deg + 90  # 坐标系逆时针旋转90度,左上角在第四象限
        approx_head_deg = np.where(head_rotated_deg < 360, head_rotated_deg, head_rotated_deg - 360)
        head_ang_order = np.argsort(approx_head_deg)[::-1]
        head_left_up = approx_head2[head_ang_order[0]]
        # 寻找 approx_tail 右上角的点
        tail_ang_order = np.argsort(approx_tail_deg)[::-1]
        tail_right_up = approx_tail2[tail_ang_order[0]]

        # 求鱼腹坐标点集
        belly_index = np.argwhere((contours_body2[:, 0] <= head_left_up[0]) &
                                  (contours_body2[:, 0] >= tail_right_up[0]) &
                                  (contours_body2[:, 1] <= tail_right_up[1]))

        belly_points = contours_body2[belly_index].reshape(-1, 2)
        # 鱼腹坐标点集按x排序
        belly_order = np.argsort(belly_points[:, 0])[::-1]
        belly_points_ordered = belly_points[belly_order]

        # 求鱼头切割点集
        head_separator_end = np.argmax(contours_head2[:, 1])
        head_points = contours_head2[0:head_separator_end + 1]
        head_points_ordered = head_points

        return head_points_ordered, belly_points_ordered


def compute(masks, class_ids, class_names, head_points_num=10, belly_points_num=20):
    mask_head = None
    mask_tail = None
    mask_body = None

    for i in range(class_ids.shape[0]):
        if class_names[class_ids[i]] == 'fish_head':
            mask_head = np.where(masks[:, :, i].astype(np.int32) == 0, 0, 255).astype(np.uint8)
        elif class_names[class_ids[i]] == 'fish_tail':
            mask_tail = np.where(masks[:, :, i].astype(np.int32) == 0, 0, 255).astype(np.uint8)
        elif class_names[class_ids[i]] == 'fish_body':
            mask_body = np.where(masks[:, :, i].astype(np.int32) == 0, 0, 255).astype(np.uint8)

    assert (mask_head is not None and mask_tail is not None and mask_body is not None)

    # 求分割线
    head_points, belly_points = head_belly_line(mask_head, mask_tail, mask_body)

    # 鱼腹分割线拟合,采样
    belly_mse, belly_samples = binomial_fit(belly_points[:, 0], belly_points[:, 1], belly_points_num)

    # 鱼头分割线拟合,采样
    head_mse, head_samples = binomial_fit(head_points[:, 1], head_points[:, 0], head_points_num)
    head_samples = head_samples[:, [1, 0]]  # 交换x,y顺序

    return head_mse, head_samples, belly_mse, belly_samples


def draw_points(sk_image, belly_points, head_points):
    cp_image = sk_image.copy()
    for point in belly_points.astype(np.int32):
        # cp_image[point[1], point[0]] = np.array([255, 255, 255])
        cv2.circle(cp_image, tuple(point), 2, (255, 255, 0), -1)

    for point in head_points.astype(np.int32):
        # cp_image[point[1], point[0]] = np.array([255, 255, 255])
        cv2.circle(cp_image, tuple(point), 2, (255, 0, 0), -1)

    return cp_image
