# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["TTVOrbit"]

import theano.tensor as tt

from .keplerian import KeplerianOrbit


class TTVOrbit(KeplerianOrbit):
    """A generalization of a Keplerian orbit with transit timing variations

    Only one of the arguments ``ttvs`` or ``transit_times`` can be given and
    the other will be computed from the one that was provided.

    Args:
        ttvs: A list (with on entry for each planet) of "O-C" vectors for each
            transit of each planet in units of days. "O-C" means the
            difference between the observed transit time and the transit time
            expected for a regular periodic orbit.
        transit_times: A list (with on entry for each planet) of transit times
            for each transit of each planet in units of days. These times will
            be used to compute the implied (least squares) ``period`` and
            ``t0`` so these parameters cannot also be given.

    """

    def __init__(self, *args, **kwargs):
        ttvs = kwargs.pop("ttvs", None)
        transit_times = kwargs.pop("transit_times", None)
        if ttvs is None and transit_times is None:
            raise ValueError(
                "one of 'ttvs' or 'transit_times' must be " "defined"
            )
        if ttvs is not None:
            self.ttvs = [tt.as_tensor_variable(ttv) for ttv in ttvs]
        else:
            if kwargs.pop("period", None) is not None:
                raise ValueError(
                    "a period cannot be given if 'transit_times' " "is defined"
                )

            self.transit_times = []
            self.ttvs = []
            period = []
            t0 = []
            for i, times in enumerate(transit_times):
                times = tt.as_tensor_variable(times)

                N = times.shape[0]
                AT = tt.stack(
                    (tt.arange(N, dtype=times.dtype), tt.ones_like(times)),
                    axis=0,
                )
                A = tt.transpose(AT)
                ATA = tt.dot(AT, A)
                ATy = tt.dot(AT, times)
                w = tt.slinalg.solve_symmetric(ATA, ATy)
                expect = tt.dot(w, AT)

                period.append(w[0])
                t0.append(w[1])
                self.ttvs.append(times - expect)
                self.transit_times.append(times)

            kwargs["period"] = tt.stack(period)
            kwargs["t0"] = tt.stack(t0)

        super(TTVOrbit, self).__init__(*args, **kwargs)
        self._base_time = 0.5 - self.t0 / self.period

        if ttvs is not None:
            self.transit_times = [
                self.t0[i] + self.period[i] * tt.arange(ttv.shape[0]) + ttv
                for i, ttv in enumerate(self.ttvs)
            ]

        # Set up a histogram for identifying the transit offsets
        self._bin_edges = [
            tt.concatenate(
                (
                    [tts[0] - 0.5 * self.period[i]],
                    0.5 * (tts[1:] + tts[:-1]),
                    [tts[-1] + 0.5 * self.period[i]],
                )
            )
            for i, tts in enumerate(self.transit_times)
        ]
        self._bin_values = [
            tt.concatenate(([0], self.ttvs[i], [0]))
            for i in range(len(self.ttvs))
        ]

    def _get_model_dt(self, t):
        vals = []
        for i in range(len(self.ttvs)):
            inds = tt.extra_ops.searchsorted(self._bin_edges[i], t)
            vals.append(self._bin_values[i][inds])
        return tt.stack(vals, -1)

    def _warp_times(self, t):
        return tt.shape_padright(t) - self._get_model_dt(t)
