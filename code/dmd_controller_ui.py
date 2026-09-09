"""
DMD Light Focusing Controller - GUI Application
================================================
Graphical user interface for controlling the DMD (Digital Micromirror Device)
and Basler camera for wavefront shaping experiments.

Features:
    - Live camera preview with interactive ROI definition
    - Load and project mask patterns from PNG files
    - Adjustable exposure (manual slider or auto-saturation)
    - Screenshot capture from live camera feed
    - Sequential phase optimisation (light focusing algorithm)
    - Save final hologram and optimal field data

Hardware:
    - DMD: ALP-4.3 API, 2560x1600 resolution
    - Camera: Basler via pypylon, hardware-triggered via DMD Line3 sync

Based on:
    I. M. Vellekoop and A. P. Mosk, Opt. Lett. 32(16), 2309-2311 (2007).
"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import time
import os
import sys
import argparse
import pandas as pd

try:
    from ALP4 import *
    import pypylon.pylon as pylon
    from phase_holograms.fields_propagation.fields import generate_target_field
    from phase_holograms.holograms.dmd_holograms import compute_lee_hologram
    from light_focusing_algorithm import (
        switch_camera_mode, measure_roi, adjust_exposure_for_saturation,
        get_segment_indices, dmd_times,
    )
    _IMPORTS_OK = True
except (ImportError, OSError, FileNotFoundError):
    _IMPORTS_OK = False

# Fallback constants when ALP4 is not available
if not _IMPORTS_OK:
    ALP_SYNCH_POLARITY = 0x0604
    ALP_LEVEL_LOW = 0x0101
    ALP_PROJ_INVERSION = 0x0313
    def dmd_times(exposure_us):
        return exposure_us + 10000, exposure_us
    def switch_camera_mode(camera, mode):
        if mode == "triggered":
            camera.StopGrabbing()
            camera.MaxNumBuffer.SetValue(8)
            camera.TriggerMode.SetValue("On")
            camera.StartGrabbing()
        else:
            camera.StopGrabbing()
            camera.TriggerMode.SetValue("Off")
            camera.StartGrabbing()
    def measure_roi(frame, roi_x, roi_y, roi_w, roi_h):
        import cv2, numpy as np
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray_roi))
    def adjust_exposure_for_saturation(camera, current_exposure_us, saturated):
        if saturated:
            new_exposure = max(current_exposure_us // 2, 5000)
            if new_exposure < current_exposure_us:
                camera.ExposureTimeAbs.SetValue(new_exposure)
        return camera.ExposureTimeAbs.GetValue()
    def get_segment_indices(seg_idx, num_seg_x, num_seg_y,
                            sq_y0=0, sq_y1=1600, sq_x0=0, sq_x1=2560):
        seg_h = (sq_y1 - sq_y0) // num_seg_y
        seg_w = (sq_x1 - sq_x0) // num_seg_x
        row = seg_idx // num_seg_x
        col = seg_idx % num_seg_x
        return (sq_y0 + row * seg_h, sq_y0 + (row + 1) * seg_h,
                sq_x0 + col * seg_w, sq_x0 + (col + 1) * seg_w)


# ------------------------------------------------------------
# Demo / mock classes for running without hardware
# ------------------------------------------------------------

CAM_WIDTH = 2448
CAM_HEIGHT = 2048
DEMO_CAM_WIDTH = 1024
DEMO_CAM_HEIGHT = 768


class _MockGrabResult:
    def __init__(self, frame):
        self._frame = frame
        self._ok = True

    def GrabSucceeded(self):
        return self._ok

    def Release(self):
        pass


class _MockConverter:
    def __init__(self):
        self._frame = None

    def Convert(self, grab_result):
        self._frame = grab_result._frame.copy()
        return self

    def GetArray(self):
        return self._frame


class _MockExposureAttr:
    def __init__(self, initial=50000):
        self._val = initial

    def GetValue(self):
        return self._val

    def get(self):
        return self._val

    def SetValue(self, val):
        self._val = val


class _MockTriggerAttr:
    def __init__(self, initial="Off"):
        self._val = initial

    def SetValue(self, val):
        self._val = val

    def GetValue(self):
        return self._val


class _MockValueAttr:
    def __init__(self, initial="Off"):
        self._val = initial

    def SetValue(self, val):
        self._val = val

    def GetValue(self):
        return self._val


class MockCamera:
    def __init__(self):
        self.ExposureTimeAbs = _MockExposureAttr(50000)
        self.ExposureAuto = _MockTriggerAttr("Off")
        self.TriggerSource = _MockTriggerAttr("Line3")
        self.TriggerMode = _MockTriggerAttr("Off")
        self.TriggerActivation = _MockTriggerAttr("RisingEdge")
        self.LineSelector = _MockTriggerAttr("Line3")
        self.LineMode = _MockTriggerAttr("Input")
        self.MaxNumBuffer = _MockValueAttr(8)
        self._grabbing = False
        self._converter = _MockConverter()
        self._phase = 0.0

    def Open(self):
        pass

    def Close(self):
        pass

    def StartGrabbing(self, *args, **kwargs):
        self._grabbing = True

    def StopGrabbing(self):
        self._grabbing = False

    def IsGrabbing(self):
        return self._grabbing

    def RetrieveResult(self, timeout_ms=100, on_timeout=None):
        import numpy as np
        time.sleep(0.02)
        self._phase += 0.3
        frame = self._make_frame()
        return _MockGrabResult(frame)

    def _make_frame(self):
        import numpy as np
        import cv2
        t = time.time()
        h, w = DEMO_CAM_HEIGHT, DEMO_CAM_WIDTH
        cx, cy = w // 2, h // 2
        y, x = np.mgrid[0:h, 0:w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        ring = np.sin(dist * 0.05 - t * 2.0 + self._phase) * 40 + 128
        ring = np.clip(ring, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(ring, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame_bgr, "DEMO MODE", (w // 2 - 80, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame_bgr


class MockDMD:
    def __init__(self):
        self._seq_counter = 0
        self._running = False

    def Initialize(self):
        pass

    def DevControl(self, *args, **kwargs):
        pass

    def ProjControl(self, *args, **kwargs):
        pass

    def SeqAlloc(self, nbImg=1, bitDepth=8):
        self._seq_counter += 1
        return self._seq_counter

    def SeqPut(self, imgData=None, SequenceId=None):
        pass

    def SetTiming(self, SequenceId=None, illuminationTime=0, pictureTime=0,
                  synchDelay=0, synchPulseWidth=1000, triggerInDelay=None):
        pass

    def Run(self, loop=False, SequenceId=None):
        self._running = True

    def Halt(self):
        self._running = False

    def FreeSeq(self, SequenceId=None):
        pass

    def Free(self):
        pass

    def Wait(self):
        pass


# Sentinel for auto-detect demo mode
_AUTO_DEMO = not _IMPORTS_OK


DMD_WIDTH = 2560
DMD_HEIGHT = 1600
NUM_PHASES_DEFAULT = 8
NUM_ROUNDS_DEFAULT = 2
DMD_SYNCH_PULSE_WIDTH = 1000
SATURATION_THRESHOLD = 240.0
MIN_EXPOSURE_US = 5000
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "gui"
)


class CameraThread(threading.Thread):
    def __init__(self, camera, converter, frame_callback, error_callback=None):
        super().__init__(daemon=True)
        self.camera = camera
        self.converter = converter
        self.frame_callback = frame_callback
        self.error_callback = error_callback
        self._running = True

    def run(self):
        while self._running:
            try:
                if _IMPORTS_OK and not isinstance(self.camera, MockCamera):
                    grabResult = self.camera.RetrieveResult(
                        100, pylon.TimeoutHandling_ThrowException
                    )
                else:
                    grabResult = self.camera.RetrieveResult(100)
                if grabResult.GrabSucceeded():
                    image = self.converter.Convert(grabResult)
                    frame = image.GetArray()
                    grabResult.Release()
                    self.frame_callback(frame)
                else:
                    grabResult.Release()
            except Exception as e:
                if self.error_callback:
                    self.error_callback(str(e))
                time.sleep(0.05)

    def stop(self):
        self._running = False
        self.join(timeout=2.0)


class OptimizationThread(threading.Thread):
    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self.cancel_event = threading.Event()
        self.latest_frame = None
        self.segment = 0
        self.total_segments = 0
        self.round_index = 0
        self.total_rounds = 0
        self.phase_idx = 0
        self.total_phases = 0
        self.phase_angle = 0.0
        self.intensity = 0.0
        self.best_phase = 0.0
        self.done = False
        self.error = None

    def run(self):
        try:
            self.app.root.after(0, lambda: self.app.log("Optimization thread started"))
            app = self.app
            DMD = app.DMD

            num_seg_x = app.grid_x.get()
            num_seg_y = app.grid_y.get()
            num_phases = app.num_phases.get()
            self.total_segments = num_seg_x * num_seg_y
            self.total_phases = num_phases

            current_field = np.ones((DMD_HEIGHT, DMD_WIDTH), dtype=complex)
            if app.loaded_field is not None:
                current_field = app.loaded_field.copy()

            current_exposure_us = app.camera.ExposureTimeAbs.GetValue()

            app.root.after(0, lambda: self.app.log(
                f"Starting optimisation: {self.total_segments} segments, "
                f"{num_phases} phases"))

            if app.camera_thread:
                app.camera_thread.stop()
                app.camera_thread = None

            try:
                app.camera.StopGrabbing()
            except Exception:
                pass
            time.sleep(0.5)

            DMD.Halt()
            DMD.DevControl(ALP_SYNCH_POLARITY, ALP_LEVEL_LOW)
            time.sleep(0.5)

            if app.seqid_display is not None:
                DMD.FreeSeq(app.seqid_display)
                app.seqid_display = None
            if app.seqid_opt is not None:
                DMD.FreeSeq(app.seqid_opt)
                app.seqid_opt = None

            app.root.after(0, lambda: self.app.log("Allocating DMD sequences..."))
            app.seqid_display = DMD.SeqAlloc(nbImg=1, bitDepth=8)
            app.seqid_opt = DMD.SeqAlloc(nbImg=num_phases, bitDepth=8)
            DMD.ProjControl(ALP_PROJ_INVERSION, 1)
            app.root.after(0, lambda: self.app.log(
                f"DMD ready: seqid_opt={app.seqid_opt}"))

            switch_camera_mode(app.camera, "triggered")
            app.root.after(0, lambda: self.app.log("Camera switched to triggered mode"))

            num_rounds = max(1, app.num_rounds.get())
            self.total_rounds = num_rounds

            t_start = time.perf_counter()
            app.root.after(0, lambda n=num_rounds: (
                self.app.log(f"Running {n} optimisation round(s)...")
            ))

            for l in range(num_rounds):
                if self.cancel_event.is_set():
                    break
                self.round_index = l
                best_intensities = []
                best_phases = []
                app.root.after(0, lambda r=l+1, n=num_rounds: (
                    self.app.log(f"--- Round {r}/{n} starting ---")
                ))
                t_round = time.perf_counter()

                for seg_idx in range(self.total_segments):
                    if self.cancel_event.is_set():
                        self.app.root.after(0, lambda: self.app.log("Optimization cancelled"))
                        break

                    self.segment = seg_idx
                    phases = np.linspace(0, 2 * np.pi, num_phases, endpoint=False)
                    r0, r1, c0, c1 = get_segment_indices(seg_idx, num_seg_x, num_seg_y)

                    t_seg = time.perf_counter()
                    app.root.after(0, lambda s=seg_idx: self.app.log(f"Computing bitmaps for segment {s}..."))
                    bitmaps = DMDControllerApp._compute_phase_bitmaps(
                        current_field, r0, r1, c0, c1, phases)
                    t_bmp = time.perf_counter()

                    app.root.after(0, lambda s=seg_idx, dt=(t_bmp-t_seg): self.app.log(
                        f"  SeqPut segment {s} {dt:.2f}s compute)..."))
                    DMD.SeqPut(imgData=bitmaps, SequenceId=app.seqid_opt)
                    picture_us, illum_us = dmd_times(current_exposure_us)
                    DMD.SetTiming(
                        SequenceId=app.seqid_opt,
                        illuminationTime=illum_us,
                        pictureTime=picture_us,
                        synchDelay=0,
                        synchPulseWidth=DMD_SYNCH_PULSE_WIDTH,
                        triggerInDelay=None,
                    )
                    app.root.after(0, lambda s=seg_idx: self.app.log(
                        f"  DMD.Run(loop=True) segment {s}..."))
                    DMD.Run(loop=True, SequenceId=app.seqid_opt)

                    intensities = []
                    for i, phase in enumerate(phases):
                        if self.cancel_event.is_set():
                            break

                        self.phase_idx = i
                        self.phase_angle = np.degrees(phase)
                        self.intensity = 0.0

                        try:
                            grabResult = app.camera.RetrieveResult(
                                5000, pylon.TimeoutHandling_ThrowException
                            )
                        except Exception:
                            app.root.after(0, lambda s=seg_idx, p=i: (
                                self.app.log(f"  WARN: camera timeout seg {s} phase {p}")
                            ))
                            intensities.append(0.0)
                            continue

                        if grabResult.GrabSucceeded():
                            image = app.converter.Convert(grabResult)
                            frame = image.GetArray()
                            grabResult.Release()

                            intens = measure_roi(
                                frame,
                                app.roi[0], app.roi[1],
                                app.roi[2], app.roi[3],
                            )
                            intensities.append(intens)
                            self.intensity = intens
                            self.latest_frame = frame.copy()

                            app.root.after(0, self._update_segment_display)
                        else:
                            grabResult.Release()
                            intensities.append(0.0)

                    DMD.Halt()

                    if self.cancel_event.is_set():
                        break

                    if intensities:
                        best_idx = np.argmax(intensities)
                        best_phase = phases[best_idx]
                        self.best_phase = np.degrees(best_phase)
                        current_field[r0:r1, c0:c1] *= np.exp(1j * best_phase)
                        best_phases.append(best_phase)
                        best_intensities.append(intensities[best_idx])

                        app.root.after(0, lambda s=seg_idx, bp=np.degrees(best_phase): (
                            self.app.log(f"  Seg {s} best = {bp:.1f} deg")
                        ))

                    if intensities and max(intensities) > SATURATION_THRESHOLD:
                        new_exposure = max(current_exposure_us // 2, MIN_EXPOSURE_US)
                        if new_exposure < current_exposure_us:
                            app.camera.ExposureTimeAbs.SetValue(new_exposure)
                            current_exposure_us = new_exposure
                            app.root.after(0, lambda e=new_exposure: (
                                self.app.log(f"  Auto-exposure: {e} us")
                            ))

                if self.cancel_event.is_set():
                    break

                elapsed = time.perf_counter() - t_round
                app.root.after(0, lambda r=l+1, n=num_rounds, e=elapsed: (
                    self.app.log(f"Round {r}/{n} complete in {e:.1f}s "
                                 f"({e/self.total_segments:.2f}s per segment)")
                ))

                os.makedirs(RESULTS_DIR, exist_ok=True)
                np.save(
                    os.path.join(RESULTS_DIR, f"optimal_field_round{l+1}.npy"),
                    current_field)
                pd.DataFrame({
                    'Phase': best_phases,
                    'Intensity': best_intensities,
                }).to_csv(
                    os.path.join(RESULTS_DIR, f"phase_intensity_round{l+1}.csv"),
                    index=False)
                app.root.after(0, lambda r=l+1: (
                    self.app.log(
                        f"Round {r}/{num_rounds} results saved to results/gui/")
                ))

            elapsed = time.perf_counter() - t_start
            app.root.after(0, lambda e=elapsed: (
                self.app.log(f"Optimisation complete in {e:.1f}s")
            ))

            switch_camera_mode(app.camera, "free_running")

            DMD.FreeSeq(app.seqid_opt)
            app.seqid_opt = None
            DMD.FreeSeq(app.seqid_display)
            app.seqid_display = DMD.SeqAlloc(nbImg=1, bitDepth=8)
            DMD.ProjControl(ALP_PROJ_INVERSION, 1)

            final_bitmap = DMDControllerApp._compute_lee_hologram_rowwise(current_field)
            DMD.SeqPut(
                imgData=final_bitmap.reshape(1, DMD_HEIGHT, DMD_WIDTH),
                SequenceId=app.seqid_display,
            )
            final_picture_us, final_illum_us = dmd_times(current_exposure_us)
            DMD.SetTiming(
                SequenceId=app.seqid_display,
                illuminationTime=final_illum_us,
                pictureTime=final_picture_us,
                synchDelay=0,
                synchPulseWidth=DMD_SYNCH_PULSE_WIDTH,
                triggerInDelay=None,
            )
            DMD.Run(loop=True, SequenceId=app.seqid_display)

            app.optimal_field = current_field
            app.final_hologram = final_bitmap

            app._start_camera_thread()

            self.done = True
            app.root.after(0, lambda: self.app.log(
                "Final hologram projected. Save results below."))

        except Exception as e:
            self.error = str(e)
            app.root.after(0, lambda: self.app.log(f"ERROR: {e}"))
            try:
                switch_camera_mode(app.camera, "free_running")
            except Exception:
                pass
            try:
                if app.seqid_opt is not None:
                    DMD.FreeSeq(app.seqid_opt)
                    app.seqid_opt = None
                if app.seqid_display is not None:
                    DMD.FreeSeq(app.seqid_display)
                app.seqid_display = DMD.SeqAlloc(nbImg=1, bitDepth=8)
                DMD.ProjControl(ALP_PROJ_INVERSION, 1)
                rec_picture_us, rec_illum_us = dmd_times(current_exposure_us)
                DMD.SetTiming(
                    SequenceId=app.seqid_display,
                    illuminationTime=rec_illum_us,
                    pictureTime=rec_picture_us,
                    synchDelay=0,
                    synchPulseWidth=DMD_SYNCH_PULSE_WIDTH,
                    triggerInDelay=None,
                )
                default_bitmap = DMDControllerApp._compute_uniform_carrier()
                DMD.SeqPut(
                    imgData=default_bitmap.reshape(1, DMD_HEIGHT, DMD_WIDTH),
                    SequenceId=app.seqid_display,
                )
                DMD.Run(loop=True, SequenceId=app.seqid_display)
            except Exception:
                pass
            try:
                app._start_camera_thread()
            except Exception:
                pass

    def _update_segment_display(self):
        if self.total_rounds > 1:
            self.app.seg_label.config(
                text=f"Round {self.round_index+1}/{self.total_rounds} | "
                     f"Seg {self.segment+1}/{self.total_segments}")
        else:
            self.app.seg_label.config(
                text=f"Seg {self.segment+1}/{self.total_segments}")
        self.app.phase_label.config(
            text=f"Phase {self.phase_idx+1}/{self.total_phases} "
                 f"({self.phase_angle:.0f} deg)")
        self.app.intensity_label.config(
            text=f"Intensity: {self.intensity:.1f}")
        total_steps = max(1, self.total_rounds * self.total_segments)
        done_steps = self.round_index * self.total_segments + self.segment
        progress = (done_steps / total_steps) * 100
        self.app.progress_var.set(progress)
        if self.latest_frame is not None:
            self.app.display_frame(self.latest_frame)


class DMDControllerApp:
    def __init__(self, root, demo_mode=False):
        self.root = root
        self.demo_mode = demo_mode or _AUTO_DEMO
        self.root.title("DMD Light Focusing Controller" + (" [DEMO MODE]" if self.demo_mode else ""))
        self.root.geometry("1200x750")
        self.root.minsize(1000, 600)

        self.camera_thread = None
        self.current_frame = None
        self._frame_lock = threading.Lock()
        self.roi = None
        self.roi_mode = False
        self.roi_start = None
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.zoom_level = 1.0
        self.zoom_cx = DMD_WIDTH // 2
        self.zoom_cy = DMD_HEIGHT // 2
        self._vis_x0 = 0
        self._vis_y0 = 0
        self._pan_start = None
        self.loaded_mask_path = None
        self.loaded_field = None
        self.optimal_field = None
        self.final_hologram = None
        self.optimization_thread = None
        self.seqid_display = None
        self.seqid_opt = None

        self._init_hardware()
        self._setup_ui()
        self._start_camera_thread()
        self._schedule_preview()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    @staticmethod
    def _compute_uniform_carrier():
        """Compute orthogonal Lee hologram for a uniform field without large temporaries.

        Uses integer arithmetic row-by-row to avoid allocating multiple
        full-frame float64 arrays.  Carrier nuvec = (1/4, 1/16).
        """
        bitmap = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
        x = np.arange(DMD_WIDTH, dtype=np.int32)
        for row in range(DMD_HEIGHT):
            val = np.mod(4 * x + row - 8, 16).astype(np.int16)
            bitmap[row] = (np.abs(val - 8) < 4).astype(np.uint8) * 255
        return bitmap

    @staticmethod
    def _compute_phase_bitmaps(current_field, r0, r1, c0, c1, phases):
        """Compute Lee hologram bitmaps for all phase offsets, row-by-row.

        Memory-efficient replacement for light_focusing_algorithm.compute_phase_bitmaps.
        Builds each test field in-place (no full 3-D copy) and computes the Lee
        hologram via _compute_lee_hologram_rowwise (one row at a time).
        """
        n_phases = len(phases)
        ny, nx = current_field.shape
        bitmaps = np.empty((n_phases, ny, nx), dtype=np.uint8)
        phase_factors = np.exp(1j * phases)
        for i in range(n_phases):
            test_field = current_field.copy()
            test_field[r0:r1, c0:c1] *= phase_factors[i]
            bitmaps[i] = DMDControllerApp._compute_lee_hologram_rowwise(test_field)
            del test_field
        return bitmaps

    @staticmethod
    def _compute_lee_hologram_rowwise(field, nuvec=(0.25, 0.0625)):
        """Compute orthogonal Lee hologram row-by-row to avoid large temporaries.

        Equivalent to compute_lee_hologram() but processes one row at a time
        so peak memory stays ~1 row instead of ~5 full-frame float64 arrays.
        """
        a_max = np.max(np.abs(field))
        if a_max == 0:
            a_max = 1.0
        nunorm = 2.0 * np.linalg.norm(nuvec)
        ux, uy = -nuvec[1] / nunorm, nuvec[0] / nunorm
        bitmap = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
        x = np.arange(DMD_WIDTH, dtype=np.float64)
        x_shifted = x - ux
        for row in range(DMD_HEIGHT):
            a_row = np.abs(field[row]) / a_max
            phi_row = np.angle(field[row])
            y_val = row + uy
            carrier = nuvec[0] * x_shifted + nuvec[1] * y_val
            cond1 = np.abs(np.mod(carrier - phi_row / (2.0 * np.pi) - 0.5, 1.0) - 0.5) < 0.25
            ortho = np.mod(-nuvec[1] * x_shifted + nuvec[0] * y_val, 1.0) < a_row
            bitmap[row] = (cond1 * ortho).astype(np.uint8) * 255
        return bitmap

    def _init_hardware(self):
        if self.demo_mode:
            self.DMD = MockDMD()
            self.camera = MockCamera()
            self.converter = self.camera._converter
            self.seqid_display = self.DMD.SeqAlloc(nbImg=1, bitDepth=8)
            self.log("DEMO MODE: Using simulated DMD and camera")
            return

        try:
            self.DMD = ALP4(
                version="4.3",
                libDir="ALP-4.3",
            )
            self.DMD.Initialize()
            self.DMD.DevControl(ALP_SYNCH_POLARITY, ALP_LEVEL_LOW)

            self.seqid_display = self.DMD.SeqAlloc(nbImg=1, bitDepth=8)
            self.DMD.ProjControl(ALP_PROJ_INVERSION, 1)

            default_bitmap = self._compute_uniform_carrier()
            self.DMD.SeqPut(
                imgData=default_bitmap.reshape(1, DMD_HEIGHT, DMD_WIDTH),
                SequenceId=self.seqid_display,
            )
            init_picture_us, init_illum_us = dmd_times(50000)
            self.DMD.SetTiming(
                SequenceId=self.seqid_display,
                illuminationTime=init_illum_us,
                pictureTime=init_picture_us,
                synchDelay=0,
                synchPulseWidth=DMD_SYNCH_PULSE_WIDTH,
                triggerInDelay=None,
            )
            self.DMD.Run(loop=True, SequenceId=self.seqid_display)

            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.camera.Open()
            self.camera.ExposureAuto.SetValue("Off")
            self.camera.ExposureTimeAbs.SetValue(50000)
            self.camera.TriggerSource.SetValue("Line3")
            self.camera.TriggerMode.SetValue("Off")
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        except Exception as e:
            print(f"Hardware init failed ({e}), falling back to demo mode")
            self.demo_mode = True
            self.root.title("DMD Light Focusing Controller [DEMO MODE]")
            self.DMD = MockDMD()
            self.camera = MockCamera()
            self.converter = self.camera._converter
            self.seqid_display = self.DMD.SeqAlloc(nbImg=1, bitDepth=8)

    def _apply_display_timing(self, exposure_us=None):
        """Set display-sequence timing to (exposure + margin, exposure)."""
        if exposure_us is None:
            exposure_us = self.exposure_var.get()
        picture_us, illum_us = dmd_times(exposure_us)
        self.DMD.SetTiming(
            SequenceId=self.seqid_display,
            illuminationTime=illum_us,
            pictureTime=picture_us,
            synchDelay=0,
            synchPulseWidth=DMD_SYNCH_PULSE_WIDTH,
            triggerInDelay=None,
        )

    def _setup_ui(self):
        style = ttk.Style()
        style.configure("Green.TButton", foreground="green")

        main_frame = ttk.Frame(self.root, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="black", cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Configure>", lambda e: self._redraw_canvas())
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", self._on_zoom)
        self.canvas.bind("<Button-5>", self._on_zoom)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_press)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<Double-Button-1>", self._on_zoom_reset)

        self.canvas_status = ttk.Label(
            canvas_frame, text="ROI: None | Camera: Initializing..."
        )
        self.canvas_status.grid(row=1, column=0, sticky="ew", pady=(3, 0))

        if self.demo_mode:
            demo_banner = tk.Label(
                canvas_frame,
                text="  DEMO MODE  --  No hardware connected. Simulated DMD and camera.  ",
                bg="#cc3300", fg="white", font=("Arial", 10, "bold"), anchor="center",
            )
            demo_banner.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        right_panel = ttk.Frame(main_frame, width=310)
        right_panel.grid(row=0, column=1, sticky="ns")
        right_panel.grid_propagate(False)
        self.right_panel = right_panel

        self._build_dmd_section(right_panel)
        self._build_beam_section(right_panel)
        self._build_exposure_section(right_panel)
        self._build_roi_section(right_panel)
        self._build_algorithm_section(right_panel)
        self._build_results_section(right_panel)

        self.log_frame = ttk.LabelFrame(self.root, text="Log", padding=3)
        self.log_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=(0, 5))
        self.log_text = tk.Text(
            self.log_frame, height=5, wrap=tk.WORD, state=tk.DISABLED
        )
        scrollbar = ttk.Scrollbar(
            self.log_frame, orient=tk.VERTICAL, command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.main_frame = main_frame

        self.fullscreen_btn = ttk.Button(
            canvas_frame, text="Fullscreen", width=10,
            command=self._toggle_fullscreen,
        )
        fs_row = 3 if self.demo_mode else 2
        self.fullscreen_btn.grid(row=fs_row, column=0, sticky="ne", pady=(2, 0))
        self._fullscreen = False

    def _build_dmd_section(self, parent):
        frame = ttk.LabelFrame(parent, text="DMD Mask", padding=5)
        frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(frame, text="Load Mask PNG", command=self._load_mask).pack(
            fill=tk.X
        )

        mask_row = ttk.Frame(frame)
        mask_row.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(mask_row, text="Loaded:").pack(side=tk.LEFT)
        self.mask_label = ttk.Label(mask_row, text="None", foreground="gray")
        self.mask_label.pack(side=tk.LEFT, padx=(3, 0))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(
            btn_row, text="Display on DMD", command=self._display_mask
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(
            btn_row, text="Clear Mask", command=self._clear_mask
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        full_row = ttk.Frame(frame)
        full_row.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(
            full_row, text="Full On", command=self._display_full_on
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(
            full_row, text="Full Off", command=self._display_full_off
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

    def _build_beam_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Beam Generator", padding=5)
        frame.pack(fill=tk.X, pady=(0, 5))

        row0 = ttk.Frame(frame)
        row0.pack(fill=tk.X)
        ttk.Label(row0, text="Type:").pack(side=tk.LEFT)
        self.beam_type_var = tk.StringVar(value="Hermite-Gauss")
        ttk.Combobox(
            row0, textvariable=self.beam_type_var, state="readonly",
            values=["Hermite-Gauss", "Laguerre-Gauss", "Speckle", "Ring"],
            width=15,
        ).pack(side=tk.LEFT, padx=(5, 0))

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(row1, text="N:").pack(side=tk.LEFT)
        self.beam_N_var = tk.IntVar(value=0)
        ttk.Spinbox(row1, from_=0, to=50, width=4,
                    textvariable=self.beam_N_var).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(row1, text="ell:").pack(side=tk.LEFT)
        self.beam_ell_var = tk.IntVar(value=0)
        ttk.Spinbox(row1, from_=-50, to=50, width=4,
                    textvariable=self.beam_ell_var).pack(side=tk.LEFT, padx=(2, 0))

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(row2, text="npw:").pack(side=tk.LEFT)
        self.beam_npw_var = tk.IntVar(value=100)
        ttk.Spinbox(row2, from_=1, to=5000, width=6,
                    textvariable=self.beam_npw_var).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(row2, text="kmax:").pack(side=tk.LEFT)
        self.beam_kmax_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(row2, from_=0.01, to=1.0, increment=0.01, width=6,
                    textvariable=self.beam_kmax_var).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(row2, text="Radius:").pack(side=tk.LEFT)
        self.beam_radius_var = tk.IntVar(value=400)
        ttk.Spinbox(row2, from_=0, to=800, width=6,
                    textvariable=self.beam_radius_var).pack(side=tk.LEFT, padx=(2, 0))

        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(row3, text="Carrier:").pack(side=tk.LEFT)
        self.beam_nu0_var = tk.DoubleVar(value=0.25)
        self.beam_nu1_var = tk.DoubleVar(value=0.0625)
        ttk.Spinbox(row3, from_=0.01, to=0.5, increment=0.01, width=5,
                    textvariable=self.beam_nu0_var).pack(side=tk.LEFT, padx=(2, 1))
        ttk.Label(row3, text=",").pack(side=tk.LEFT)
        ttk.Spinbox(row3, from_=0.0, to=0.5, increment=0.01, width=5,
                    textvariable=self.beam_nu1_var).pack(side=tk.LEFT, padx=(1, 0))

        self.project_btn = ttk.Button(
            frame, text="Project Beam", command=self._project_beam
        )
        self.project_btn.pack(fill=tk.X, pady=(5, 2))

        self.beam_status_label = ttk.Label(frame, text="", foreground="gray")
        self.beam_status_label.pack(anchor=tk.W)

    def _project_beam(self):
        import threading as _thread_mod
        self.project_btn.config(state=tk.DISABLED)
        self.beam_status_label.config(text="Generating...", foreground="blue")
        self.log("Project beam: started")

        def _worker():
            try:
                field_type_map = {
                    "Hermite-Gauss": "hermite",
                    "Laguerre-Gauss": "laguerre",
                    "Speckle": "speckle",
                    "Ring": "ring",
                }
                field_type = field_type_map[self.beam_type_var.get()]
                N = self.beam_N_var.get()
                ell = self.beam_ell_var.get()
                npw = self.beam_npw_var.get()
                kmax = self.beam_kmax_var.get()
                radius = self.beam_radius_var.get()

                if field_type == "ring":
                    cy, cx = DMD_HEIGHT // 2, DMD_WIDTH // 2
                    radius_sq = radius * radius
                    bitmap = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
                    x = np.arange(DMD_WIDTH, dtype=np.int32)
                    dx2 = (x - cx) ** 2
                    for row in range(DMD_HEIGHT):
                        dy2 = (row - cy) ** 2
                        inside = dx2 + dy2 <= radius_sq
                        carrier = np.mod(4 * x + row - 8, 16).astype(np.int16)
                        on = (np.abs(carrier - 8) < 4) & inside
                        bitmap[row] = on.astype(np.uint8) * 255

                    self.loaded_field = None
                    self.DMD.Halt()
                    self.DMD.SeqPut(
                        imgData=bitmap.reshape(1, DMD_HEIGHT, DMD_WIDTH),
                        SequenceId=self.seqid_display,
                    )
                    self.DMD.ProjControl(ALP_PROJ_INVERSION, 1)
                    self._apply_display_timing()
                    self.DMD.Run(loop=True, SequenceId=self.seqid_display)

                    self.root.after(0, lambda: (
                        self.beam_status_label.config(
                            text=self.beam_type_var.get(), foreground="green"),
                        self.project_btn.config(state=tk.NORMAL),
                        self.log(f"Beam projected: Ring r={radius}"),
                    ))
                    return
                elif field_type == "speckle":
                    from phase_holograms.fields_propagation.fields import speckle_gauss
                    y_arr = np.arange(DMD_HEIGHT)
                    x_arr = np.arange(DMD_WIDTH)
                    xx, yy = np.meshgrid(x_arr, y_arr)
                    field = speckle_gauss(npw, kmax, xx, yy)
                else:
                    field = generate_target_field(
                        field_type, Nt=N, ell=ell,
                        nx=DMD_WIDTH, ny=DMD_HEIGHT,
                    )

                nuvec = (self.beam_nu0_var.get(), self.beam_nu1_var.get())
                from phase_holograms.holograms.dmd_holograms import orthogonal_lee
                hologram = orthogonal_lee(field, nuvec=nuvec, renorm=True)
                hologram = hologram.astype(np.uint8) * 255

                self.loaded_field = field
                self.DMD.Halt()
                self.DMD.SeqPut(
                    imgData=hologram.reshape(1, DMD_HEIGHT, DMD_WIDTH),
                    SequenceId=self.seqid_display,
                )
                self.DMD.ProjControl(ALP_PROJ_INVERSION, 1)
                self._apply_display_timing()
                self.DMD.Run(loop=True, SequenceId=self.seqid_display)

                self.root.after(0, lambda: (
                    self.beam_status_label.config(
                        text=self.beam_type_var.get(), foreground="green"),
                    self.project_btn.config(state=tk.NORMAL),
                    self.log(f"Beam projected: {self.beam_type_var.get()} "
                             + (f"r={radius}" if field_type == "ring"
                                else f"N={N} ell={ell}")),
                ))
            except Exception as e:
                self.root.after(0, lambda: (
                    self.beam_status_label.config(
                        text=f"Error: {e}", foreground="red"),
                    self.project_btn.config(state=tk.NORMAL),
                    self.log(f"Beam projection failed: {e}"),
                ))

        _thread_mod.Thread(target=_worker, daemon=True).start()

    def _build_exposure_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Exposure", padding=5)
        frame.pack(fill=tk.X, pady=(0, 5))

        self.auto_exposure_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="Auto-exposure (saturation-based)",
            variable=self.auto_exposure_var,
            command=self._toggle_auto_exposure,
        ).pack(anchor=tk.W)

        slider_row = ttk.Frame(frame)
        slider_row.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(slider_row, text="Manual:").pack(side=tk.LEFT)

        self.exposure_var = tk.IntVar(value=50000)
        self.exposure_scale = ttk.Scale(
            slider_row,
            from_=1000,
            to=200000,
            variable=self.exposure_var,
            orient=tk.HORIZONTAL,
            command=self._on_exposure_change,
        )
        self.exposure_scale.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 5))

        self.exposure_entry_var = tk.StringVar(value="50000")
        self.exposure_entry = ttk.Entry(
            slider_row, textvariable=self.exposure_entry_var, width=8,
        )
        self.exposure_entry.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(slider_row, text="us").pack(side=tk.LEFT)
        self.exposure_entry.bind("<Return>", self._on_exposure_entry)
        self.exposure_entry.bind("<FocusOut>", self._on_exposure_entry)

    def _build_roi_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Region of Interest", padding=5)
        frame.pack(fill=tk.X, pady=(0, 5))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X)
        ttk.Button(
            btn_row, text="Draw ROI", command=self._enter_roi_mode
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(
            btn_row, text="Clear ROI", command=self._clear_roi
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.roi_info_label = ttk.Label(frame, text="ROI: None")
        self.roi_info_label.pack(anchor=tk.W, pady=(3, 0))

    def _build_algorithm_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Algorithm", padding=5)
        frame.pack(fill=tk.X, pady=(0, 5))

        grid_row = ttk.Frame(frame)
        grid_row.pack(fill=tk.X)
        ttk.Label(grid_row, text="Grid:").pack(side=tk.LEFT)
        self.grid_x = tk.IntVar(value=9)
        self.grid_y = tk.IntVar(value=9)
        ttk.Spinbox(
            grid_row, from_=1, to=50, width=4, textvariable=self.grid_x
        ).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Label(grid_row, text="x").pack(side=tk.LEFT)
        ttk.Spinbox(
            grid_row, from_=1, to=50, width=4, textvariable=self.grid_y
        ).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(grid_row, text="Phases:").pack(side=tk.LEFT)
        self.num_phases = tk.IntVar(value=NUM_PHASES_DEFAULT)
        ttk.Spinbox(
            grid_row, from_=2, to=32, width=4, textvariable=self.num_phases
        ).pack(side=tk.LEFT, padx=(5, 0))

        ttk.Label(grid_row, text="Rounds:").pack(side=tk.LEFT)
        self.num_rounds = tk.IntVar(value=NUM_ROUNDS_DEFAULT)
        ttk.Spinbox(
            grid_row, from_=1, to=10, width=3, textvariable=self.num_rounds
        ).pack(side=tk.LEFT, padx=(5, 0))

        self.run_btn = ttk.Button(
            frame, text="Run Optimisation", command=self._run_optimization
        )
        self.run_btn.pack(fill=tk.X, pady=(5, 3))

        self.cancel_btn = ttk.Button(
            frame, text="Cancel", command=self._cancel_optimization,
            state=tk.DISABLED,
        )
        self.cancel_btn.pack(fill=tk.X, pady=(0, 3))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill=tk.X)

        status_row = ttk.Frame(frame)
        status_row.pack(fill=tk.X, pady=(3, 0))
        self.seg_label = ttk.Label(status_row, text="Seg: -")
        self.seg_label.pack(side=tk.LEFT)
        self.phase_label = ttk.Label(status_row, text="Phase: -")
        self.phase_label.pack(side=tk.LEFT, padx=(8, 0))
        self.intensity_label = ttk.Label(status_row, text="Intensity: -")
        self.intensity_label.pack(side=tk.LEFT, padx=(8, 0))

    def _build_results_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Results", padding=5)
        frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(
            frame, text="Save Hologram PNG", command=self._save_hologram
        ).pack(fill=tk.X, pady=(0, 2))
        ttk.Button(
            frame, text="Save Field (.npy)", command=self._save_field
        ).pack(fill=tk.X, pady=(0, 2))
        ttk.Button(
            frame, text="Save Screenshot", command=self._save_screenshot
        ).pack(fill=tk.X)

    def _start_camera_thread(self):
        self.camera_thread = CameraThread(
            self.camera, self.converter,
            self._on_frame_captured, self._on_camera_error,
        )
        self.camera_thread.start()

    def _on_frame_captured(self, frame):
        with self._frame_lock:
            self.current_frame = frame.copy()

    def _on_camera_error(self, msg):
        self.root.after(0, lambda m=msg: self.log(f"Camera error: {m}"))

    def _schedule_preview(self):
        with self._frame_lock:
            frame = self.current_frame
        if frame is not None:
            self.display_frame(frame)
            if self.canvas_status.cget("text") == "ROI: None | Camera: Initializing...":
                self.canvas_status.config(text="ROI: None | Camera: Live")
        self.root.after(33, self._schedule_preview)

    def _toggle_fullscreen(self):
        if not self._fullscreen:
            self.right_panel.grid_forget()
            self.log_frame.pack_forget()
            self.fullscreen_btn.config(text="Exit Fullscreen")
            self.root.bind("<Escape>", lambda e: self._toggle_fullscreen())
            self._fullscreen = True
        else:
            self.log_frame.pack(
                fill=tk.BOTH, expand=False, padx=5, pady=(0, 5),
                after=self.main_frame,
            )
            self.right_panel.grid(row=0, column=1, sticky="ns")
            self.fullscreen_btn.config(text="Fullscreen")
            self.root.unbind("<Escape>")
            self._fullscreen = False

    def display_frame(self, frame):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        h, w = frame.shape[:2]

        vis_w = w / self.zoom_level
        vis_h = h / self.zoom_level
        x0 = int(self.zoom_cx - vis_w / 2)
        y0 = int(self.zoom_cy - vis_h / 2)
        x1 = x0 + int(vis_w)
        y1 = y0 + int(vis_h)

        pad_l = max(0, -x0)
        pad_t = max(0, -y0)
        x0c = max(0, x0)
        y0c = max(0, y0)
        x1c = min(w, x1)
        y1c = min(h, y1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop = frame_rgb[y0c:y1c, x0c:x1c]

        if pad_l > 0 or pad_t > 0:
            padded = np.zeros_like(frame_rgb[:int(vis_h), :int(vis_w)])
            padded[pad_t:pad_t + crop.shape[0], pad_l:pad_l + crop.shape[1]] = crop
            crop = padded

        crop_h, crop_w = crop.shape[:2]
        scale = min(cw / crop_w, ch / crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)

        self.display_scale = scale
        self.display_offset_x = (cw - new_w) // 2
        self.display_offset_y = (ch - new_h) // 2
        self._vis_x0 = x0c
        self._vis_y0 = y0c

        pil_img = Image.fromarray(crop).resize(
            (new_w, new_h), Image.Resampling.LANCZOS
        )

        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            draw = ImageDraw.Draw(pil_img)
            sx = int((rx - x0c) * scale)
            sy = int((ry - y0c) * scale)
            sw = int(rw * scale)
            sh = int(rh * scale)
            draw.rectangle(
                [sx, sy, sx + sw, sy + sh], outline="lime", width=2
            )
            cx, cy = sx + sw // 2, sy + sh // 2
            draw.line([(cx - 6, cy), (cx + 6, cy)], fill="lime", width=1)
            draw.line([(cx, cy - 6), (cx, cy + 6)], fill="lime", width=1)

        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.display_offset_x,
            self.display_offset_y,
            anchor=tk.NW,
            image=self._photo,
        )

        if self._fullscreen:
            self.canvas.create_text(
                cw // 2, 15,
                text="ESC to exit fullscreen",
                fill="yellow",
                font=("Arial", 11, "bold"),
            )

        if self.zoom_level > 1.0:
            zoom_txt = f"Zoom: {self.zoom_level:.1f}x"
            self.canvas.create_text(
                cw - 10, 15, anchor=tk.E,
                text=zoom_txt, fill="cyan",
                font=("Arial", 10, "bold"),
            )

        if self.demo_mode:
            self.canvas.create_text(
                cw // 2, ch - 15,
                text="DEMO MODE -- No hardware connected",
                fill="#ff6600",
                font=("Arial", 10, "bold"),
            )

        if self.roi_mode:
            self.canvas.create_text(
                cw // 2,
                15,
                text="CLICK & DRAG to set ROI  |  ESC to cancel",
                fill="yellow",
                font=("Arial", 11, "bold"),
            )

    def _redraw_canvas(self):
        with self._frame_lock:
            frame = self.current_frame
        if frame is not None:
            self.display_frame(frame)

    def _on_zoom(self, event):
        with self._frame_lock:
            frame = self.current_frame
        if frame is None:
            return
        h, w = frame.shape[:2]

        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = 1.25
        else:
            factor = 1 / 1.25

        new_zoom = self.zoom_level * factor
        new_zoom = max(1.0, min(new_zoom, 32.0))
        if new_zoom == self.zoom_level:
            return

        cam_x = (event.x - self.display_offset_x) / self.display_scale + self._vis_x0
        cam_y = (event.y - self.display_offset_y) / self.display_scale + self._vis_y0

        new_vis_w = w / new_zoom
        new_vis_h = h / new_zoom
        self.zoom_cx = max(new_vis_w / 2, min(cam_x, w - new_vis_w / 2))
        self.zoom_cy = max(new_vis_h / 2, min(cam_y, h - new_vis_h / 2))
        self.zoom_level = new_zoom

    def _on_pan_press(self, event):
        self._pan_start = (event.x, event.y, self.zoom_cx, self.zoom_cy)

    def _on_pan_drag(self, event):
        if self._pan_start is None:
            return
        sx, sy, scx, scy = self._pan_start
        dx = (event.x - sx) / self.display_scale
        dy = (event.y - sy) / self.display_scale
        self.zoom_cx = scx - dx
        self.zoom_cy = scy - dy

        with self._frame_lock:
            frame = self.current_frame
        if frame is not None:
            h, w = frame.shape[:2]
            vis_w = w / self.zoom_level
            vis_h = h / self.zoom_level
            self.zoom_cx = max(vis_w / 2, min(self.zoom_cx, w - vis_w / 2))
            self.zoom_cy = max(vis_h / 2, min(self.zoom_cy, h - vis_h / 2))

    def _on_zoom_reset(self, event):
        self.zoom_level = 1.0
        with self._frame_lock:
            frame = self.current_frame
        if frame is not None:
            h, w = frame.shape[:2]
            self.zoom_cx = w // 2
            self.zoom_cy = h // 2

    def _on_canvas_motion(self, event):
        with self._frame_lock:
            frame = self.current_frame
        if frame is None:
            return
        h, w = frame.shape[:2]
        px = int((event.x - self.display_offset_x) / self.display_scale + self._vis_x0)
        py = int((event.y - self.display_offset_y) / self.display_scale + self._vis_y0)
        if 0 <= px < w and 0 <= py < h:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            intensity = int(gray[py, px])
            roi_text = f"ROI: {self.roi if self.roi else 'None'}"
            self.canvas_status.config(
                text=f"{roi_text} | Camera: Live | "
                     f"Cursor: ({px},{py}) Intensity: {intensity}"
            )
        else:
            roi_text = f"ROI: {self.roi if self.roi else 'None'}"
            self.canvas_status.config(
                text=f"{roi_text} | Camera: Live"
            )

    def _on_canvas_press(self, event):
        if self.roi_mode:
            self.roi_start = (event.x, event.y)

    def _on_canvas_drag(self, event):
        if self.roi_mode and self.roi_start is not None:
            self.canvas.delete("roi_temp")
            self.canvas.create_rectangle(
                self.roi_start[0],
                self.roi_start[1],
                event.x,
                event.y,
                outline="yellow",
                width=2,
                dash=(4, 4),
                tags="roi_temp",
            )

    def _on_canvas_release(self, event):
        if not self.roi_mode or self.roi_start is None:
            return

        x1, y1 = self.roi_start
        x2, y2 = event.x, event.y

        ix1 = int((min(x1, x2) - self.display_offset_x) / self.display_scale + self._vis_x0)
        iy1 = int((min(y1, y2) - self.display_offset_y) / self.display_scale + self._vis_y0)
        ix2 = int((max(x1, x2) - self.display_offset_x) / self.display_scale + self._vis_x0)
        iy2 = int((max(y1, y2) - self.display_offset_y) / self.display_scale + self._vis_y0)

        with self._frame_lock:
            frame = self.current_frame
        if frame is not None:
            fh, fw = frame.shape[:2]
        else:
            fw, fh = DMD_WIDTH, DMD_HEIGHT

        ix1 = max(0, min(ix1, fw))
        iy1 = max(0, min(iy1, fh))
        ix2 = max(0, min(ix2, fw))
        iy2 = max(0, min(iy2, fh))

        w = ix2 - ix1
        h = iy2 - iy1

        if w > 0 and h > 0:
            self.roi = (ix1, iy1, w, h)
            self.roi_info_label.config(
                text=f"ROI: ({ix1}, {iy1}) {w}x{h}"
            )
            self.canvas_status.config(
                text=f"ROI: ({ix1},{iy1}) {w}x{h} | Camera: Live"
            )
            self.log(f"ROI set: ({ix1}, {iy1}) {w}x{h}")
        else:
            self.log("ROI too small, ignored")

        self.roi_mode = False
        self.roi_start = None
        self.canvas.delete("roi_temp")
        self.canvas.configure(cursor="")

    def _enter_roi_mode(self):
        self.roi_mode = True
        self.roi_start = None
        self.canvas.configure(cursor="crosshair")
        self.log("ROI mode: click and drag on the canvas")
        self.root.bind("<Escape>", self._cancel_roi_mode)

    def _cancel_roi_mode(self, event=None):
        self.roi_mode = False
        self.roi_start = None
        self.canvas.delete("roi_temp")
        self.canvas.configure(cursor="")
        self.root.unbind("<Escape>")
        self.log("ROI mode cancelled")

    def _clear_roi(self):
        self.roi = None
        self.roi_info_label.config(text="ROI: None")
        self.canvas_status.config(text="ROI: None | Camera: Live")
        self.log("ROI cleared")

    def _load_mask(self):
        filepath = filedialog.askopenfilename(
            title="Load Mask PNG",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if not filepath:
            return

        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            self.log(f"Error: Could not load {filepath}")
            return

        mask = cv2.resize(mask, (DMD_WIDTH, DMD_HEIGHT))
        phase_pattern = mask.astype(np.float64) / 255.0 * 2 * np.pi
        field = np.exp(1j * phase_pattern)

        self.loaded_mask_path = filepath
        self.loaded_field = field
        self.mask_label.config(
            text=os.path.basename(filepath), foreground="black"
        )
        self.log(f"Mask loaded: {os.path.basename(filepath)}")

    def _display_mask(self):
        if self.loaded_field is None:
            self.log("No mask loaded. Use 'Load Mask PNG' first.")
            return

        hologram = DMDControllerApp._compute_lee_hologram_rowwise(self.loaded_field)
        self.DMD.Halt()
        self.DMD.SeqPut(
            imgData=hologram.reshape(1, DMD_HEIGHT, DMD_WIDTH),
            SequenceId=self.seqid_display,
        )
        self.DMD.ProjControl(ALP_PROJ_INVERSION, 1)
        self._apply_display_timing()
        self.DMD.Run(loop=True, SequenceId=self.seqid_display)
        self.log("Mask displayed on DMD")

    def _clear_mask(self):
        self.loaded_field = None
        self.loaded_mask_path = None
        self.mask_label.config(text="None", foreground="gray")

        default_bitmap = self._compute_uniform_carrier()
        self.DMD.Halt()
        self.DMD.SeqPut(
            imgData=default_bitmap.reshape(1, DMD_HEIGHT, DMD_WIDTH),
            SequenceId=self.seqid_display,
        )
        self.DMD.ProjControl(ALP_PROJ_INVERSION, 1)
        self._apply_display_timing()
        self.DMD.Run(loop=True, SequenceId=self.seqid_display)
        self.log("Mask cleared, default hologram restored")

    def _display_full_on(self):
        bitmap = self._compute_uniform_carrier()
        self.DMD.Halt()
        self.DMD.SeqPut(
            imgData=bitmap.reshape(1, DMD_HEIGHT, DMD_WIDTH),
            SequenceId=self.seqid_display,
        )
        self.DMD.ProjControl(ALP_PROJ_INVERSION, 1)
        self._apply_display_timing()
        self.DMD.Run(loop=True, SequenceId=self.seqid_display)
        self.log("Full On: all mirrors reflecting")

    def _display_full_off(self):
        bitmap = np.full((DMD_HEIGHT, DMD_WIDTH), 255, dtype=np.uint8)
        self.DMD.Halt()
        self.DMD.SeqPut(
            imgData=bitmap.reshape(1, DMD_HEIGHT, DMD_WIDTH),
            SequenceId=self.seqid_display,
        )
        self.DMD.ProjControl(ALP_PROJ_INVERSION, 1)
        self._apply_display_timing()
        self.DMD.Run(loop=True, SequenceId=self.seqid_display)
        self.log("Full Off: all mirrors dark")

    def _toggle_auto_exposure(self):
        if self.auto_exposure_var.get():
            self.exposure_scale.config(state=tk.DISABLED)
            self.exposure_entry.config(state=tk.DISABLED)
            self.log("Auto-exposure enabled")
        else:
            self.exposure_scale.config(state=tk.NORMAL)
            self.exposure_entry.config(state=tk.NORMAL)
            val = self.exposure_var.get()
            self.camera.ExposureTimeAbs.SetValue(val)
            self._apply_display_timing(val)
            self.log(f"Manual exposure: {val} us")

    def _on_exposure_change(self, value):
        if not self.auto_exposure_var.get():
            val = int(float(value))
            self.camera.ExposureTimeAbs.SetValue(val)
            self.exposure_entry_var.set(str(val))
            self._apply_display_timing(val)

    def _on_exposure_entry(self, event=None):
        if self.auto_exposure_var.get():
            return
        try:
            val = int(self.exposure_entry_var.get())
            val = max(1000, min(val, 200000))
        except ValueError:
            val = self.exposure_var.get()
        self.exposure_entry_var.set(str(val))
        self.exposure_var.set(val)
        self.camera.ExposureTimeAbs.SetValue(val)
        self._apply_display_timing(val)

    def _run_optimization(self):
        if self.roi is None:
            messagebox.showwarning(
                "No ROI", "Please define a Region of Interest first."
            )
            return

        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.seg_label.config(text="Seg: -")
        self.phase_label.config(text="Phase: -")
        self.intensity_label.config(text="Intensity: -")

        self.optimization_thread = OptimizationThread(self)
        self.optimization_thread.start()
        self._poll_optimization()

    def _poll_optimization(self):
        t = self.optimization_thread
        if t is None:
            return

        if t.done or t.error:
            self.run_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
            self.progress_var.set(100 if t.done else 0)
            self.seg_label.config(text="Seg: Done" if t.done else "Seg: Error")
            if t.error:
                self.log(f"Optimization failed: {t.error}")
                messagebox.showerror("Optimization Error", t.error)
            self.optimization_thread = None
            return

        self.root.after(200, self._poll_optimization)

    def _cancel_optimization(self):
        if self.optimization_thread and not self.optimization_thread.done:
            self.optimization_thread.cancel_event.set()
            self.log("Cancellation requested...")

    def _save_screenshot(self):
        with self._frame_lock:
            frame = self.current_frame
        if frame is None:
            self.log("No frame to save")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Screenshot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("TIFF", "*.tif")],
            initialfile=f"screenshot_{int(time.time())}.png",
        )
        if filepath:
            cv2.imwrite(filepath, frame)
            self.log(f"Screenshot saved: {os.path.basename(filepath)}")

    def _save_hologram(self):
        if self.final_hologram is None:
            self.log("No hologram to save. Run optimisation first.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Hologram",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp")],
            initialfile="final_hologram.png",
        )
        if filepath:
            cv2.imwrite(filepath, self.final_hologram)
            self.log(f"Hologram saved: {os.path.basename(filepath)}")

    def _save_field(self):
        if self.optimal_field is None:
            self.log("No field data to save. Run optimisation first.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Optimal Field",
            defaultextension=".npy",
            filetypes=[("NumPy", "*.npy")],
            initialfile="optimal_field.npy",
        )
        if filepath:
            np.save(filepath, self.optimal_field)
            self.log(f"Field saved: {os.path.basename(filepath)}")

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        prefix = f"[{timestamp}] {message}\n"
        try:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, prefix)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        except AttributeError:
            print(prefix, end="")

    def _on_close(self):
        if self.optimization_thread and not self.optimization_thread.done:
            if not messagebox.askokcancel(
                "Quit", "Optimization is running. Quit anyway?"
            ):
                return
            self.optimization_thread.cancel_event.set()

        if self.camera_thread:
            self.camera_thread.stop()

        if not self.demo_mode:
            try:
                self.DMD.Halt()
                if self.seqid_display:
                    self.DMD.FreeSeq(self.seqid_display)
                if self.seqid_opt:
                    self.DMD.FreeSeq(self.seqid_opt)
                self.DMD.Free()
            except Exception:
                pass

            try:
                self.camera.StopGrabbing()
                self.camera.Close()
            except Exception:
                pass

        cv2.destroyAllWindows()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="DMD Light Focusing Controller GUI"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode without hardware (simulated DMD and camera)",
    )
    args, _ = parser.parse_known_args()

    root = tk.Tk()
    app = DMDControllerApp(root, demo_mode=args.demo)
    root.mainloop()


if __name__ == "__main__":
    main()
