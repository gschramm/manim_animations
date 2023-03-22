from manim import *
import numpy as np

def read_GE_ak_wav(fname: str):
    N = {}
    offset = 0

    desc = np.fromfile(fname, dtype=np.int8, offset=offset, count=256)
    offset += desc.size * desc.itemsize

    N["gpts"] = np.fromfile(fname,
                            dtype=np.dtype('>u2'),
                            offset=offset,
                            count=1)[0]
    offset += N["gpts"].size * N["gpts"].itemsize

    N["groups"] = np.fromfile(fname,
                              dtype=np.dtype('>u2'),
                              offset=offset,
                              count=1)[0]
    offset += N["groups"].size * N["groups"].itemsize

    N["intl"] = np.fromfile(fname,
                            dtype=np.dtype('>u2'),
                            offset=offset,
                            count=N["groups"])
    offset += N["intl"].size * N["intl"].itemsize

    N["params"] = np.fromfile(fname,
                              dtype=np.dtype('>u2'),
                              offset=256 + 4 + N["groups"] * 2,
                              count=1)[0]
    offset += N["params"].size * N["params"].itemsize

    params = np.fromfile(fname,
                         dtype=np.dtype('>f8'),
                         offset=offset,
                         count=N["params"])
    offset += params.size * params.itemsize

    wave = np.fromfile(fname, dtype=np.dtype('>i2'), offset=offset)
    offset += wave.size * wave.itemsize

    grad = np.swapaxes(wave.reshape((N["groups"], N["intl"][0], N["gpts"])), 0,
                       2)

    # set stop bit to 0
    grad[-1, ...] = 0

    # scale gradients to SI units (T/m)
    grad = (grad / 32767) * (params[3] / 100)

    # bandwidth in (Hz)
    bw = 1e6 / params[7]

    # (proton) field of view in (m)
    fov = params[1] / 100

    return grad, bw, fov, desc, N, params

#--------------------------------------------------------------------------------

class TPIScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        axes = ThreeDAxes(x_range=(-2,2,1), y_range=(-2,2,1), z_range=(-2,2,1))
        self.add(axes)

        self.begin_ambient_camera_rotation(rate=-0.5)
        self.wait(0.5)

        gradient_file = '/Users/georg/Library/CloudStorage/Box-Box/SodiumData/20230316_MR3_GS_QED/pfiles/g16/ak_grad56.wav'
        grads_T_m, bw, fov, desc, N, params = read_GE_ak_wav(gradient_file)
        dt_us = params[7]
        gamma_by_2pi_MHz_T: float = 11.262
        k_1_cm = 0.01 * np.cumsum(grads_T_m, axis=0) * dt_us * gamma_by_2pi_MHz_T

        for i in np.arange(0, k_1_cm.shape[1], 8):
            curve = ParametricFunction(
                lambda u: k_1_cm[int(u),i,:], color=RED, t_range = (0,k_1_cm.shape[0]-1,8)).set_shade_in_3d(True)

            self.play(Create(curve, run_time=0.4))
            curve.set_stroke(opacity=0.15)
            self.add(Dot3D(k_1_cm[-1,i,:], radius = 0.02))
            self.wait(0.1)
            #self.play(FadeOut(curve))
