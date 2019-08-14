import numpy as np
import random
import pyglet
from math import *
from random import gauss, seed
import matplotlib.pyplot as plt
from gym import spaces


LENGTH = 120

top_a = np.pi / 6.
btm_a = np.pi * 5 / 6.

""" 
    The cam parameters to simulate the real case
"""
cam_info = {'fx': 572.4110, 'fy': 573.5704,
            'cx': 325.26, 'cy': 242.04899, 'w': 640, 'h': 480}
cam_R = [0, 0, 0]
cam_T = np.array([[0], [0], [0]])


"""
 At the beginning of each episode, the angle between spreader and container
 is set within range as follows:
  - angle_bound provides big range
  - p_angle_bound provides the range for micro-motion contol
"""
angle_bound = [-1 / 3 * np.pi + 0.02, 1 / 3 * np.pi + 0.02]
p_angle_bound = [-0.7 / 18. * np.pi, 0.7 / 18. * np.pi]

""" 
    4 types of spreader-container noise 
"""
no_noise = {'xy': 0., 'angle': 0., 'h': 0.}
small = {'xy': 0.06, 'angle': 1.2 / 180 * np.pi, 'h': 0.1}
middle = {'xy': 0.12, 'angle': 2.2 / 180 * np.pi, 'h': 0.6}
big = {'xy': 0.25, 'angle': 3.2 / 180 * np.pi, 'h': 1.2}
hardest = {'xy': 0.6, 'angle': 5.2 / 180 * np.pi, 'h': 1.5}

SC_NOISE = [small, middle, big, hardest, no_noise]


"""
    noise on camera's  parameters
"""


def add_cam_noises(rotation=False, translation=False, RT=False, intrisic=False, addAll=False):
    R_n = [np.radians(gauss(0, 2)), np.radians(
        gauss(0, 2)), np.radians(gauss(0, 2))]
    T_n = np.array([[gauss(0, 0.12)], [gauss(0, 0.12)], [gauss(0, 0.12)]])

    intrinsic_n = [gauss(0, 4), gauss(0, 4)]
    if rotation:
        return R_n
    elif translation:
        return T_n
    elif RT:
        return R_n, T_n
    elif intrisic:
        return intrinsic_n
    elif addAll:
        return R_n, T_n, intrinsic_n
    return

"""
    noise on detected corner coordinates
"""


def add_xy_noises():
    mu, sigma = 0., 4
    xn = np.random.normal(mu, sigma, 1)[0]
    yn = np.random.normal(mu, sigma, 1)[0]
    return [xn, yn]


def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return trunc(stepper * number) / stepper


def angle_between_vec(v0, v1):
    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    if angle > 0:
        angle = (np.pi - abs(angle))
    else:
        angle = -1 * (np.pi - abs(angle))
    return angle


def symmetric_op(point, s_type, unnormalize=False, cx=cam_info['cx'], cy=cam_info['cy'], w=cam_info['w'], h=cam_info['h']):
    p = point
    w = 1
    h = 1

    if s_type == "r2l_f":  # right to left
        p[0] = w - p[0]
    elif s_type == "l2r_f":  # left to right
        p[0] = 2 * p[0] - w
    elif s_type == "d2u_f":  # down to up
        p[1] = h - p[1]
    elif s_type == "u2d_f":  # up to down
        p[1] = 2 * p[1] - h
    else:
        print("no such type")

    if unnormalize:
        p[0] = p[0] * cam_info['w']
        p[1] = p[1] * cam_info['h']

    return p


"""camera projection pinhole model"""


def e2p(euclidean):
    assert(type(euclidean) == np.ndarray)
    assert((euclidean.shape[0] == 3) | (euclidean.shape[0] == 2))

    # print(euclidean.shape[1])
    return np.vstack((euclidean, np.ones((1, euclidean.shape[1]))))


def euler2rot(euler_angles):
    cx, cy, cz = np.cos(euler_angles)
    sx, sy, sz = np.sin(euler_angles)
    Rx = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    Ry = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    Rz = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])

    # R = Rx*Ry*Rz
    return np.dot(Rx, np.dot(Ry, Rz))


def set_p_para(K, R, t):
    return K.dot(np.hstack((self.R, self.t)))


def get_k_para(add_in_noise=False):
    # using Camera resectioning matrix https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters
    # values from
    # http://ptak.felk.cvut.cz/6DB/public/datasets/hinterstoisser/camera.yml
    in_noise = add_xy_noises()
    if add_in_noise:
        fx = cam_info['fx'] + in_noise[0]
        fy = cam_info['fy'] + in_noise[1]
    else:
        fx = cam_info['fx']  # + np.random.normal(0,.1,1)
        fy = cam_info['fy']  # + np.random.normal(0,.1,1)

    cx = cam_info['cx']  # + np.random.normal(0,.1,1)
    cy = cam_info['cy']  # + np.random.normal(0,.1,1)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def proj2img_v0(point):
    centre_x = cam_info['cx']
    centre_y = cam_info['cy']
    # using Camera resectioning matrix https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters
    # set the camera intrinsic matrix
    K = get_k_para()
    # set the camera ratation angles
    euler_angles = [np.radians(gauss(0, 0)), np.radians(
        gauss(0, 0)), np.radians(gauss(0, 0))]
    # transer to the camera ratation matrix (extrinsic matrix)
    R = euler2rot(euler_angles)
    # seed(1)
    t = np.array([[0], [0], [0]])
    # convert the relative coordinates (container's corner to spreader)
    # print(point)
    # print(np.hstack((R,t)))
    point = e2p(point)
    cam_coord = np.hstack((R, t)).dot(point)

    xy = cam_coord[0:2, :]
    z = cam_coord[2:]
    image_coords_distorted_metrics = xy / z
    res = K.dot(e2p(image_coords_distorted_metrics)).T[0][:2]
    print(res)
    coords = []
    for idx, r in enumerate(res):
        # print(r)
        if idx == 0:
            coords.append(truncate((r - centre_x) * 1. / centre_x, 5))
        if idx == 1:
            coords.append(truncate((r - centre_y) * 1. / centre_y, 5))

    return coords


def proj2img(point, normalize=True, noise_r=False, noise_t=False, noise_in=False, test=False):  # noise_r 2.0, noise_t 0.12
    import camera
    c = camera.Camera()
    # set the camera ratation angles
    euler_angles = cam_R

    T = cam_T

    c.set_K(get_k_para(False))

    if noise_in and (not test):
        c.set_K(get_k_para(True))

    if noise_r and (not test):
        euler_angles = add_cam_noises(rotation=True)
        # print(euler_angles)
    if noise_t and (not test):
        T = add_cam_noises(translation=True)
        # print(T)

    # transer to the camera ratation matrix (extrinsic matrix)
    R = euler2rot(euler_angles)
    c.set_R(R)
    c.set_t(T)
    point = np.hstack(point)
    pixel_c = np.hstack(c.world_to_image(
        np.array([[point[0], point[1], point[2]]]).T))
    #print(pixel_c, point[2])
    coords = []

    if normalize:
        for idx, p in enumerate(pixel_c):
            # print(r)
            if idx == 0:
                coords.append(truncate((p * 1. / cam_info['w']), 5))
            if idx == 1:
                coords.append(truncate((p * 1. / cam_info['h']), 5))

        return coords
    else:
        return [pixel_c[0], pixel_c[1]]


class c_env(object):
    viewer = None
    reach = 0
    dt = 0.1    # refresh rate
    dxy = 9  # value should be 0.5 for projection micro_motion
    action_bound = [-1, 1]
    goal = {'l': LENGTH, 'r': 0., 'cx': 200, 'cy': 200}
    p_paras = {'offset': 25, 'h': 2.1}
    state_dim = 4
    use_corner = True
    action_dim = 3
    projection = False
    angle_bound = None
    noise = SC_NOISE[0]
    """
    # to plot the camera coordinates
    fig = plt.figure()
    plt.ion()

    tl_fig = fig.add_subplot(221)
    tr_fig = fig.add_subplot(222)
    br_fig = fig.add_subplot(223)
    bl_fig = fig.add_subplot(224)
    """

    cam_noise_r = False
    cam_noise_t = False
    cam_noise_in = False

    test_model = False

    action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)

    def __init__(self, obs_mode,
                 noise_mode="small",
                 reward_type="2_corners",
                 projection=False,
                 cam_R_noise=False,
                 cam_T_noise=False,
                 cam_IN_noise=False,
                 test=False):

        self.mode = obs_mode
        self.noise_mode = noise_mode
        self.reward_type = reward_type
        self.projection = projection

        self.cam_noise_r = cam_R_noise
        self.cam_noise_t = cam_T_noise
        self.cam_noise_in = cam_IN_noise

        self.test_model = test
        if projection:
            # print('proj------------')
            self.angle_bound = p_angle_bound
            if obs_mode == 'easy':
                self.state_dim = 3
            elif obs_mode == '2_corners':
                self.state_dim = 4
            elif obs_mode == '4_corners':
                self.state_dim = 17
            elif obs_mode == 'all':
                self.state_dim = 12
            elif obs_mode == 'partial_all':
                self.state_dim = 8
            elif obs_mode == "symmetric_corners":
                self.state_dim = 12
            elif obs_mode == "no_symmetric_corners":
                self.state_dim = 14
            elif obs_mode == "no_symmetric_corners_simple":
                self.state_dim = 10
            else:
                print("no such mode")
        else:
            self.angle_bound = angle_bound

            if obs_mode == 'easy':
                self.state_dim = 3
            elif obs_mode == '2_corners':
                self.state_dim = 4
            elif obs_mode == '4_corners':
                self.state_dim = 8
            elif obs_mode == 'all':
                self.state_dim = 11
            elif obs_mode == 'partial_all':
                self.state_dim = 7
            else:
                print("no such mode")

        if noise_mode == "small":
            self.noise = SC_NOISE[0]
        elif noise_mode == "middle":
            self.noise = SC_NOISE[1]
        elif noise_mode == "big":
            self.noise = SC_NOISE[2]
        elif noise_mode == "hardest":
            self.noise = SC_NOISE[3]
        elif noise_mode == "none":
            self.noise = SC_NOISE[4]
        else:
            print("wrong noise type")

        print(self.noise)

        self.crane_info = np.zeros(
            1, dtype=[('l', np.float32), ('r', np.float32), ('cx', np.float32), ('cy', np.float32)])
        self.crane_info['l'] = LENGTH
        self.crane_info['r'] = np.pi / 6
        self.crane_info['cx'] = 100
        self.crane_info['cy'] = 100

    def step(self, action):
        reach_c = 20
        done = False
        r = 0.
        action = np.clip(action, *self.action_bound)
        for idx, a in enumerate(action):
            # print(idx)
            if abs(action[idx]) > 0.1:
                a = action[idx]
            else:
                a = 0
            action[idx] = a
        self.crane_info['r'] += action[0] * self.dt
        # self.crane_info['r'] %= np.pi * 2    # normalize
        self.crane_info['cx'] += action[1] + random.uniform(-1 * 0.12, 0.12)
        self.crane_info['cy'] += action[2] + random.uniform(-1 * 0.12, 0.12)

        if self.crane_info['cx'] < 50:
            self.crane_info['cx'] = 50
        if self.crane_info['cy'] < 50:
            self.crane_info['cy'] = 50
        if self.crane_info['cx'] > 350:
            self.crane_info['cx'] = 350
        if self.crane_info['cy'] > 350:
            self.crane_info['cy'] = 350

        if self.crane_info['r'] < self.angle_bound[0]:
            self.crane_info['r'] = self.angle_bound[0]
        if self.crane_info['r'] > self.angle_bound[1]:
            self.crane_info['r'] = self.angle_bound[1]

        """
        self.crane_info['r'] = 0
        # self.crane_info['r'] %= np.pi * 2    # normalize
        self.crane_info['cx'] = 190
        self.crane_info['cy'] = 201

        self.goal['r'] = 0
        # self.crane_info['r'] %= np.pi * 2    # normalize
        self.goal['cx'] = 200
        self.goal['cy'] = 200
        """

        # state
        # s = self.crane_info['r']

        s, gt = self._get_state(self.mode, projection=self.projection)
        # print(s)
        r_mode = self.reward_type

        if r_mode == '2_corners':
            sr, __ = self._get_state(r_mode, projection=False)
            r = self._r_func(sr, r_mode)
            if (r * -1 < 0.04):
                print(r)
                r += 1.
                self.reach += 1
                if self.reach > reach_c:
                    done = True
                    r = 200
            else:
                #r = -1
                self.reach = 0
                done = False

        elif r_mode == 'no_symmetric_corners':
            sr, __ = self._get_state(r_mode, projection=True)
            r = self._r_func(sr, r_mode)
            if (r * -1 < 0.04):
                print(r)
                r += 1.
                self.reach += 1
                if self.reach > reach_c:
                    done = True
                    r = 200
            else:
                #r = -1
                self.reach = 0
                done = False

        elif r_mode == 'easy':
            sr, __ = self._get_state(r_mode, projection=False)
            r = self._r_func(sr, r_mode)
            if (abs(sr[0]) < 0.05 and abs(sr[1]) < 0.05 and abs(sr[2]) / np.pi * 180) < 1:
                r += 1.
                self.reach += 1
                if self.reach > reach_c:
                    done = True
                    r = 200
            else:
                #r = -1
                self.reach = 0
                done = False
        # print(r)
        return s, r, done, gt

    def _get_state(self, mode='easy', projection=False):
        br = self.crane_info['r']  # radian, angle
        bcx = self.crane_info['cx']  # radian, angle
        bcy = self.crane_info['cy']  # radian, angle
        gr = self.goal['r']  # radian, angle
        gcx = self.goal['cx']  # radian, angle
        gcy = self.goal['cy']  # radian, angle
        da = (br % np.pi - gr % np.pi) * 180
        sa = (br % np.pi) * 180

        dx = (gcx - bcx) / 100.
        dy = (gcy - bcy) / 100.

        s_c = np.hstack(get_coords(self.crane_info))
        c_c = np.hstack(get_coords(self.goal))

        sc00 = np.array([s_c[0], s_c[1]])
        sc01 = np.array([s_c[2], s_c[3]])
        sc10 = np.array([s_c[4], s_c[5]])
        sc11 = np.array([s_c[6], s_c[7]])

        cc00 = np.array([c_c[0], c_c[1]])
        cc01 = np.array([c_c[2], c_c[3]])
        cc10 = np.array([c_c[4], c_c[5]])
        cc11 = np.array([c_c[6], c_c[7]])

        d00 = cc00 - sc00
        d01 = cc01 - sc01
        d10 = cc10 - sc10
        d11 = cc11 - sc11

        gt = [dx, dy, da]

        if projection:
            #print('pro:', self.mode)
            img_coords = []
            h = self.p_paras['h']
            tl = np.hstack((d00, h))
            tr = np.hstack((d01, h))
            br = np.hstack((d10, h))
            bl = np.hstack((d11, h))

            points = [tl, tr, br, bl]
            # print(points)
            for p in points:
                #print(p[0], p[1])
                p = [[p[0]], [p[1]], [p[2]]]
               # print(p)
                xy = proj2img(
                    np.array([p[0][0] / 100., p[1][0] / 100., p[2]]),
                    normalize=True,
                    noise_r=self.cam_noise_r,
                    noise_t=self.cam_noise_t,
                    noise_in=self.cam_noise_in,
                    test=self.test_model)

                #print(self.cam_noise_r, self.cam_noise_t, self.cam_noise_in)
                # tmp = proj2img(
                #    np.array([p[0][0] / 100., p[1][0] / 100., p[2]]), normalize=False)
                #print("not normalized", tmp)

                img_coords.append(xy[0])
                img_coords.append(xy[1])

            # print(points)
            # print(img_coords)
            #print("tl:", img_coords[0]*centre_x, img_coords[1]*centre_y)
            #print("tr:", img_coords[2]*centre_x, img_coords[3]*centre_y)
            #print("br:", img_coords[4]*centre_x, img_coords[5]*centre_y)
            #print("bl:", img_coords[6]*centre_x, img_coords[7]*centre_y)
            # print("----------------------")

            if mode == '2_corners':
                # print('yes')
                im_c = np.hstack((img_coords[0], img_coords[1], img_coords[
                    4], img_coords[5]))
                s = np.hstack((im_c))
                #s = np.hstack((im_c, s_c[0], s_c[1], s_c[4], s_c[5], h))
                # print(s)

            elif mode == '4_corners':
                #s = np.hstack((np.array(img_coords)))
                s = np.hstack((np.array(img_coords), s_c, h))
                # print(len(s))

            elif mode == 'symmetric_corners':
                p_tl = np.array([img_coords[0], img_coords[1]])
                p_tr = np.array([img_coords[2], img_coords[3]])
                p_br = np.array([img_coords[4], img_coords[5]])
                p_bl = np.array([img_coords[6], img_coords[7]])
                tl2tr = symmetric_op(p_tl, s_type="l2r_f")  # flip tl to tr
                d_tl_tr = p_tr - tl2tr
                bl2br = symmetric_op(p_bl, s_type="l2r_f")  # flip bl to br
                d_bl_br = p_br - bl2br
                tl2bl = symmetric_op(p_tl, s_type="u2d_f")  # flip tl to bl
                d_tl_bl = p_bl - tl2bl
                tr2br = symmetric_op(p_tr, s_type="u2d_f")  # flip bl to br
                d_tr_br = p_br - tr2br

                tl2br = symmetric_op(symmetric_op(
                    p_tl, s_type="l2r_f"), s_type="u2d_f")  # flip tl to br
                d_tl_br = p_br - tl2br
                tr2bl = symmetric_op(symmetric_op(
                    p_tl, s_type="l2r_f"), s_type="u2d_f")  # flip tl to br
                d_tr_bl = p_bl - tr2bl

                s = np.hstack((d_tl_tr, d_bl_br, d_tl_bl,
                               d_tr_br, d_tl_br, d_tr_bl))
                print("symmetric_op:", s)
                # print(len(s))

            elif mode == 'no_symmetric_corners' or mode == 'no_symmetric_corners_simple':

                p_tl = np.array([img_coords[0], img_coords[1]])
                p_tr = np.array([img_coords[2], img_coords[3]])
                p_br = np.array([img_coords[4], img_coords[5]])
                p_bl = np.array([img_coords[6], img_coords[7]])

                d_tl_tr = p_tr - p_tl

                d_bl_br = p_br - p_bl

                d_tl_bl = p_bl - p_tl

                d_tr_br = p_br - p_tr

                d_tl_br = p_br - p_tl

                d_tr_bl = p_bl - p_tr

               # print("d_tl_tr", d_tl_tr)

                # point to 4 image plane's common center
                v_tl_c = np.array([1 - p_tl[0], 1 - p_tl[1]])
                v_tr_c = np.array([-1 * p_tr[0], 1 - p_tr[1]])
                v_br_c = np.array([-1 * p_br[0], -1 * p_br[1]])
                v_bl_c = np.array([1 - p_bl[0], -1 * p_bl[1]])

               # print(v_tl_c, v_tr_c, v_br_c, v_bl_c)

                """
                distance_tl = np.linalg.norm(v_tl_c)
                distance_tr = np.linalg.norm(v_tr_c)c
                distance_br = np.linalg.norm(v_br_c)
                distance_bl = np.linalg.norm(v_bl_c)

                dis_tl_br = distance_tl - distance_br
                dis_tl_tr = distance_tl - distance_tr
                dis_tr_bl = distance_tr - distance_bl
                dis_bl_br = distance_bl - distance_br
                """

                angle_tl_br = angle_between_vec(v_tl_c, v_br_c) * 5
                angle_tr_bl = angle_between_vec(v_tr_c, v_bl_c) * 5

                if mode == 'no_symmetric_corners':
                    s = np.hstack((d_tl_tr, d_bl_br, d_tl_bl,
                                   d_tr_br, d_tl_br, d_tr_bl, angle_tl_br, angle_tr_bl))
                else:
                    s = np.hstack(
                        (v_tl_c, v_tr_c, v_br_c, v_bl_c, angle_tl_br, angle_tr_bl))

                #print("symmetric_op:", s)
                # print(len(s))

            elif mode == 'easy':
                # noise in meters, 0.01=1cm
                dx += random.uniform(-1 * self.noise['xy'], self.noise['xy'])
                # noise in meters, 0.01=1cm
                dy += random.uniform(-1 * self.noise['xy'], self.noise['xy'])
                # noise in radius
                da += random.uniform(-1 *
                                     self.noise['angle'], self.noise['angle'])
                s = np.hstack((dx, dy, da))

            elif mode == 'all':
                # noise in meters, 0.01=1cm
                dx += random.uniform(-1 * self.noise['xy'], self.noise['xy'])
                # noise in meters, 0.01=1cm
                dy += random.uniform(-1 * self.noise['xy'], self.noise['xy'])
                # noise in radius
                da += random.uniform(-1 *
                                     self.noise['angle'], self.noise['angle'])
                h += random.uniform(-1 * self.noise['h'], self.noise['h'])

                s = np.hstack((np.array(img_coords) / 100., dx, dy, da, h))

            elif mode == 'partial_all':
                # noise in meters, 0.01=1cm
                dx += random.uniform(-1 * self.noise['xy'], self.noise['xy'])
                # noise in meters, 0.01=1cm
                dy += random.uniform(-1 * self.noise['xy'], self.noise['xy'])
                # noise in radius
                da += random.uniform(-1 *
                                     self.noise['angle'], self.noise['angle'])
                sa += random.uniform(-1 *
                                     self.noise['angle'], self.noise['angle'])

                h += random.uniform(-1 * self.noise['h'], self.noise['h'])

                an = 0
                angle_difference = True
                if angle_difference:
                    an = da
                else:
                    an = sa

                s = np.hstack((img_coords[0], img_coords[1], img_coords[
                              4], img_coords[5], dx, dy, an, h))
            else:
                print("no such mode")
                return None

            return s, gt

        else:

            if mode == '2_corners':
                s = np.hstack((d00, d10)) / 100.

            elif mode == '4_corners':
                s = np.hstack((d00, d01, d10, d11)) / 100.

            elif mode == 'easy':
                s = np.hstack((dx, dy, da))

            elif mode == 'all':
                s = np.hstack((d00, d01, d10, d11, dx, dy, da))

            elif mode == 'partial_all':
                s = np.hstack((d00, d10, dx, dy, da))

            else:
                print("no such mode")
                return None

            # print("s", s)
            return s, gt

    def _r_func(self, obs,  mode):
        if mode == '2_corners' or mode == 'symmetric_corners' or mode == 'no_symmetric_corners':
            r = np.linalg.norm(obs) * -1
            # print(r)
        elif mode == 'easy':
            r = -1 * np.sum(np.absolute(obs))
        return r

    def reset(self):
        small_reset = True
        if self.projection or small_reset:
            self.dxy = 0.3  # + random.uniform(-0.2, .2)
            self.dt = 0.15  # + random.uniform(-0.2, .2)
            ar = 0.
            self.crane_info['r'] = ar
            self.crane_info['cx'] = 200
            self.crane_info['cy'] = 200
            gr = random.uniform(p_angle_bound[0], p_angle_bound[1])
            self.goal['r'] = gr
            # print(ar, gr)
            self.goal[
                'cx'] = 200 + random.uniform(-1 * self.p_paras['offset'], self.p_paras['offset'])
            self.goal[
                'cy'] = 200 + random.uniform(-1 * self.p_paras['offset'], self.p_paras['offset'])

            self.p_paras['h'] = random.uniform(1.7, 3.5)
            reach = 0
            done = False
            s, gt = self._get_state(self.mode, projection=self.projection)
        else:
            ar = random.uniform(angle_bound[0], angle_bound[1])
            self.crane_info['r'] = ar
            self.crane_info['cx'] = random.uniform(90, 310)
            self.crane_info['cy'] = random.uniform(90, 310)
            gr = random.uniform(angle_bound[0], angle_bound[1])
            self.goal['r'] = gr
            # print(ar, gr)
            self.goal['cx'] = random.uniform(90, 310)
            self.goal['cy'] = random.uniform(90, 310)
            reach = 0
            done = False
            s, gt = self._get_state(self.mode)
        return s, gt

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.crane_info, self.goal)
        self.viewer.render()

    def sample_action(self):
        # two radians
        dx = random.uniform(-1., 1.0)
        dy = random.uniform(-1., 1.0)
        r = random.uniform(-1., 1.0)
        # print(dx, dy)
        return [r - 0.5, dx, dy]


class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, crane_info, goal_info):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400,
                                     resizable=False, caption='crane', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.crane_info = crane_info
        self.center_coord = np.array([200, 200])
        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal_info = goal_info
        goal_coords = get_coords(self.goal_info)
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', goal_coords),
            ('c3B', (86, 109, 249) * 4))    # color

        self.r_box = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 120,              # location
                     100, 180,
                     200, 120,
                     200, 180]), ('c3B', (249, 86, 86) * 4,))

        self.center_p = [0, 0]

    def render(self):
        self._update_crane()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_crane(self):

        # spreader
        self.goal.vertices = get_coords(self.goal_info)
        self.r_box.vertices = get_coords(self.crane_info)


def get_coords(info):
        # container
    cx = info['cx']
    cy = info['cy']
    cp = np.array([cx, cy])
    # cp = [200, 200]
    bl = info['l']
    ar = info['r']  # - info['r'] + angle_bound[0]
    #ar = a
    xy_tl0 = np.array(
        [np.cos(top_a * -1 + ar), np.sin(top_a * -1 + ar)]) * bl + cp
    xy_tr0 = np.array(
        [np.cos(top_a + ar), np.sin(top_a + ar)]) * bl + cp
    xy_bl0 = np.array(
        [np.cos(btm_a * -1 + ar), np.sin(btm_a * -1 + ar)]) * bl + cp
    xy_br0 = np.array(
        [np.cos(btm_a + ar), np.sin(btm_a + ar)]) * bl + cp

    return np.concatenate((xy_tl0, xy_tr0, xy_br0, xy_bl0))

if __name__ == '__main__':
    print('crane')
    # env = craneEnv()
    # env.reset()
    # while True:
    #    env.render()
    #    env.step(env.sample_action())
