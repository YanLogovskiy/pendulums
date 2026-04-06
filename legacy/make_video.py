from matplotlib import pylab as plt
from num_integration_methods import rungekut
from models import model_decorator
import os
import math


def create_frame_oscillation(t, x, w, b):
    plt.clf()
    plt.plot([-3, 3], [0, 0], '-k')
    plt.plot(x[0], 0, 'ob', ms=20)
    plt.plot([0, x[0]], [0, 0], '-r', ms=10)

    plt.xlim([-3, 3])
    plt.title('t = {time:.2f} c,  omega = {omega}, dissip_coeff = {betta}'.format(time=t, omega=w, betta=b))
    plt.savefig('frames/f_{:04.0f}.png'.format(t * 100), bbox_inches='tight', dpi=240)


def create_frame_math_pendulum(t, a, w, b):
    plt.clf()
    plt.plot([-3, 3], [0, 0], '-k')
    x = 2 * math.sin(a[0])
    y = - 2 * math.cos(a[0])
    plt.plot(x, y, 'ob', ms=20)
    plt.plot([0, x], [0, y], '-r', ms=10)

    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.title('t = {time:.2f} c,  omega = {omega}, dissip_coeff = {betta}'.format(time=t, omega=w, betta=b))
    plt.savefig('frames/f_{:04.0f}.png'.format(t * 100), bbox_inches='tight', dpi=240)


def create_video(properties, model_func, create_frame_func):
    duration = properties[2]
    frame_per_second = 25
    num_of_frames = int(duration * frame_per_second)
    step = 1 / frame_per_second

    if not os.path.exists('frames'):
        os.makedirs('frames')
    os.system('rm frames/*')
    if not os.path.exists('videos'):
        os.makedirs('videos')
    os.system('rm videos/*')

    t = 0
    x = [properties[3], properties[4]]
    w = properties[0]
    b = properties[1]

    model_func = model_decorator([w, b])(model_func)
    create_frame_func(t, x, w, b)
    for i in range(num_of_frames):
        x = rungekut(model_func, t, t + step, x)
        t = t + step
        create_frame_func(t, x, w, b)

    video_name = 'videos/simulation_{name}_omega_{omega}_dissip_{betta}.avi'.format(
            name=model_func.__name__, omega=w, betta=b)
    if os.path.exists(video_name):
        os.remove(video_name)
    os.system('ffmpeg -f image2 -pattern_type glob -framerate 25 -i \"frames/f_*.png\" '
              '-s 1080x1080 {video_name}'.format(video_name=video_name))

