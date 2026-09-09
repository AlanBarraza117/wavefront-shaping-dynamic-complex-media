"""
Light Focusing Through Complex Media - Sequential Phase Optimisation
====================================================================
Implementation of the iterative wavefront shaping algorithm described in:

    I. M. Vellekoop and A. P. Mosk,
    "Focusing coherent light through opaque strongly scattering media,"
    Optics Letters, vol. 32, no. 16, pp. 2309-2311, Aug. 2007.
    https://doi.org/10.1364/OL.32.002309
    Repository: https://github.com/AlanBarraza117/DMD-Controller-for-light-focusing-through-complex-media

The algorithm divides the DMD wavefront into segments, tests a set of phase
offsets (0 to 2pi) on each segment one at a time, and keeps the phase that
maximises the camera intensity at a target ROI. The process is repeated
sequentially for every segment, building up an optimised wavefront that
constructively interferes at the target point behind the scattering medium.

Hardware:
    - DMD (Digital Micromirror Device): ALP-4.3 API, 2560x1600 resolution
    - Camera: Basler (via pypylon), hardware-triggered via DMD Line3 sync

Optimisations over naive implementation:
    - Batch bitmap pre-computation via numpy broadcasting (vectorised fields)
    - All phase bitmaps uploaded as a single multi-frame DMD sequence
    - Camera hardware-triggered by DMD sync signal (no time.sleep)
    - Persistent DMD sequence allocation (no FreeSeq/SeqAlloc per segment)
    - Reduced DMD frame time (100ms vs 200ms) matched to camera exposure
    - Camera stays in triggered mode throughout optimisation (no mode switches)

This script was developed as part of a summer internship project focused on
adaptive optics and wavefront shaping for imaging through scattering media.
"""
from ALP4 import *
import cv2
import pypylon.pylon as pylon
import time
import numpy as np
from phase_holograms.fields_propagation.fields import generate_target_field
from phase_holograms.holograms.dmd_holograms import compute_lee_hologram
import pandas as pd
import os


# ------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------

NUM_PHASES = 8                         # Phase values to test per segment
DMD_SYNCH_PULSE_WIDTH = 1000           # Sync pulse width to camera (us)
DMD_EXPOSURE_MARGIN_US = 10000         # Safety margin: pictureTime = exposure + this
SATURATION_THRESHOLD = 240.0           # ROI intensity above this triggers exposure reduction
MIN_EXPOSURE_US = 5000                 # Floor for auto-exposure reduction
PROGRESS_INTERVAL = 5                  # Show interim hologram every N segments (0 = never)


def dmd_times(exposure_us):
    """Return (pictureTime, illuminationTime) in microseconds for the given exposure.

    pictureTime must exceed exposure_us so the camera finishes readout
    before the next trigger fires.
    """
    picture = exposure_us + DMD_EXPOSURE_MARGIN_US
    return picture, exposure_us


# ------------------------------------------------------------
# Camera mode management
# ------------------------------------------------------------

def switch_camera_mode(camera, mode):
    """
    Switches the Basler camera between free-running and hardware-triggered modes.

    During live view and debug, the camera runs in free-running mode
    (TriggerMode=Off, GrabStrategy_LatestImageOnly). During optimisation
    measurement, it switches to hardware-triggered mode (TriggerMode=On,
    TriggerSource=Line3, GrabStrategy_OneByOne) so each DMD frame
    transition triggers exactly one camera capture.

    Parameters
    ----------
    camera : pylon.InstantCamera
        Basler camera instance.
    mode : str
        'free_running' or 'triggered'.
    """
    if mode == 'triggered':
        camera.StopGrabbing()
        camera.MaxNumBuffer.SetValue(NUM_PHASES)
        camera.LineSelector.SetValue("Line3")
        camera.LineMode.SetValue("Input")
        camera.TriggerMode.SetValue("On")
        camera.TriggerSource.SetValue("Line3")
        camera.TriggerActivation.SetValue("RisingEdge")
        camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    elif mode == 'free_running':
        camera.StopGrabbing()
        camera.TriggerMode.SetValue("Off")
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


# ------------------------------------------------------------
# Measurement helpers
# ------------------------------------------------------------

def measure_roi(frame, roi_x, roi_y, roi_w, roi_h):
    """
    Measures the average grayscale intensity within a region of interest (ROI).

    Uses the mean instead of max so a single hot pixel does not dominate
    the feedback signal, giving a much more stable optimisation target.

    Parameters
    ----------
    frame : ndarray
        Camera frame in BGR format.
    roi_x, roi_y : int
        Top-left corner of the ROI in pixel coordinates.
    roi_w, roi_h : int
        Width and height of the ROI in pixels.

    Returns
    -------
    float
        Average grayscale intensity within the ROI.
    """
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray_roi))


def adjust_exposure_for_saturation(camera, current_exposure_us, saturated):
    """
    Reduces camera exposure time if the ROI is saturated.

    When the ROI intensity reaches near the maximum 8-bit value (255),
    the phase optimisation becomes unreliable because multiple phases
    may all read the same maximum. This function halves the exposure
    time to bring the intensity back into a usable dynamic range.

    Parameters
    ----------
    camera : pylon.InstantCamera
        Basler camera instance.
    current_exposure_us : int
        Current exposure time in microseconds.
    saturated : bool
        True if the ROI was saturated during the last segment sweep.

    Returns
    -------
    int
        Updated exposure time in microseconds.
    """
    if saturated:
        new_exposure = max(current_exposure_us // 2, MIN_EXPOSURE_US)
        if new_exposure < current_exposure_us:
            camera.ExposureTimeAbs.SetValue(new_exposure)
            print(f"\n  [AUTO-EXPOSURE] ROI saturated! Reducing exposure: "
                  f"{current_exposure_us} -> {new_exposure} us")
        else:
            print(f"\n  [AUTO-EXPOSURE] ROI saturated but already at minimum exposure "
                  f"({MIN_EXPOSURE_US} us). Consider reducing DMD illumination time.")
    return camera.ExposureTimeAbs.GetValue()


# ------------------------------------------------------------
# Segment helpers
# ------------------------------------------------------------

def get_segment_indices(seg_idx, num_seg_x, num_seg_y,
                        sq_y0=0, sq_y1=1600, sq_x0=0, sq_x1=2560):
    """
    Computes the row/column boundaries for a given segment index.

    Divides the square region (sq_y0:sq_y1, sq_x0:sq_x1) into a grid
    of num_seg_x * num_seg_y segments and returns the absolute DMD
    pixel coordinates for the requested segment.

    Parameters
    ----------
    seg_idx : int
        Linear index of the segment (0-based, row-major).
    num_seg_x, num_seg_y : int
        Number of segments along x and y axes.
    sq_y0, sq_y1 : int
        Row boundaries of the square region on the DMD.
    sq_x0, sq_x1 : int
        Column boundaries of the square region on the DMD.

    Returns
    -------
    tuple of int
        (row_start, row_end, col_start, col_end) absolute pixel indices.
    """
    seg_h = (sq_y1 - sq_y0) // num_seg_y
    seg_w = (sq_x1 - sq_x0) // num_seg_x
    row = seg_idx // num_seg_x
    col = seg_idx % num_seg_x
    return (sq_y0 + row * seg_h, sq_y0 + (row + 1) * seg_h,
            sq_x0 + col * seg_w, sq_x0 + (col + 1) * seg_w)


def compute_phase_bitmaps(current_field, r0, r1, c0, c1, phases, nuvec=(0.25, 0.0625)):
    """
    Pre-computes Lee hologram bitmaps for all phase offsets.

    Processes one phase at a time with row-by-row hologram computation
    to avoid allocating large temporary arrays (no n_phases stack, no meshgrids).

    Parameters
    ----------
    current_field : ndarray (complex)
        The cumulative complex field (2D, shape ny x nx).
    r0, r1, c0, c1 : int
        Pixel boundaries of the segment being optimised.
    phases : ndarray
        Array of phase offsets to test (radians).
    nuvec : tuple
        Carrier frequency vector (f_x, f_y) for the Lee hologram.
        Default (0.25, 0.0625) matches the DMD controller UI default.

    Returns
    -------
    ndarray (uint8)
        Stack of bitmaps, shape (N, ny, nx).
    """
    n_phases = len(phases)
    ny, nx = current_field.shape

    nunorm = 2 * np.sqrt(nuvec[0]**2 + nuvec[1]**2)
    x_off = -nuvec[1] / nunorm
    y_off = nuvec[0] / nunorm

    x = np.arange(nx, dtype=np.float64) + x_off

    a_full = np.abs(current_field)
    max_a = np.max(a_full)
    if max_a > 0:
        a_norm = a_full / max_a
    else:
        a_norm = a_full
    phi_full = np.angle(current_field)

    bitmaps = np.empty((n_phases, ny, nx), dtype=np.uint8)

    seg_backup = current_field[r0:r1, c0:c1].copy()

    for i in range(n_phases):
        phase = phases[i]
        current_field[r0:r1, c0:c1] = seg_backup * np.exp(1j * phase)

        a_row_full = np.abs(current_field)
        if max_a > 0:
            a_row_norm = a_row_full / np.max(a_row_full)
        else:
            a_row_norm = a_row_full
        phi_row = np.angle(current_field)

        for row in range(ny):
            y_eff = row + y_off
            term1 = nuvec[0]*x + nuvec[1]*y_eff - phi_row[row]/(2*np.pi) - 0.5
            cond1 = np.abs(np.mod(term1, 1.0) - 0.5) < 0.25

            term2 = -nuvec[1]*x + nuvec[0]*y_eff
            cond2 = np.mod(term2, 1.0) < a_row_norm[row]

            bitmaps[i, row] = (cond1 & cond2).astype(np.uint8) * 255

    current_field[r0:r1, c0:c1] = seg_backup

    return bitmaps


def compute_lee_hologram_row_by_row(field, nuvec=(1/4, 1/16)):
    """
    Computes an orthogonal Lee hologram bitmap using row-by-row processing
    to avoid allocating large meshgrid arrays.

    Parameters
    ----------
    field : ndarray (complex)
        The target complex field (2D, shape ny x nx).
    nuvec : tuple
        Carrier frequency vector.

    Returns
    -------
    ndarray (uint8)
        Binary Lee hologram bitmap.
    """
    ny, nx = field.shape
    a = np.abs(field)
    max_a = np.max(a)
    if max_a > 0:
        a /= max_a
    phi = np.angle(field)

    nunorm = 2 * np.sqrt(nuvec[0]**2 + nuvec[1]**2)
    x_off = -nuvec[1] / nunorm
    y_off = nuvec[0] / nunorm

    x = np.arange(nx, dtype=np.float64) + x_off

    bitmap = np.empty((ny, nx), dtype=np.uint8)
    for row in range(ny):
        y_eff = row + y_off
        term1 = nuvec[0]*x + nuvec[1]*y_eff - phi[row]/(2*np.pi) - 0.5
        cond1 = np.abs(np.mod(term1, 1.0) - 0.5) < 0.25
        term2 = -nuvec[1]*x + nuvec[0]*y_eff
        cond2 = np.mod(term2, 1.0) < a[row]
        bitmap[row] = (cond1 & cond2).astype(np.uint8) * 255

    return bitmap


# ------------------------------------------------------------
# Core optimisation function
# ------------------------------------------------------------

def optimize_segment(DMD, camera, converter, seqid, seg_idx, roi_x, roi_y, roi_w, roi_h,
                     current_field, num_seg_x, num_seg_y,
                     sq_y0=0, sq_y1=1600, sq_x0=0, sq_x1=2560, display_seqid=None,
                     exposure_us=140000):
    """
    Optimises the phase offset for a single segment of the DMD.

    Fully optimised measurement cycle:
        1. Pre-computes all phase bitmaps via vectorised construction
        2. Overwrites the persistent DMD sequence with SeqPut (no alloc/free)
        3. Camera is already in hardware-triggered mode (set before the loop)
        4. DMD plays sequence once (loop=False) -- each frame triggers camera
        5. Grabs exactly NUM_PHASES frames with zero time.sleep
        6. Displays each frame in real-time with ROI and intensity overlay

    Parameters
    ----------
    DMD : ALP4
        ALP4 DMD controller instance.
    camera : pylon.InstantCamera
        Basler camera instance (must already be in triggered mode).
    converter : pylon.ImageFormatConverter
        Converter for camera image format.
    seqid : int
        Persistent DMD sequence ID (overwritten each call, never freed here).
    seg_idx : int
        Index of the segment being optimised.
    roi_x, roi_y, roi_w, roi_h : int
        ROI coordinates on the camera frame.
    current_field : ndarray (complex)
        The cumulative complex field; modified in-place with the best phase.
    num_seg_x, num_seg_y : int
        Grid division of the DMD into segments.

    Returns
    -------
    tuple of (float, bool)
        The best phase offset (radians) found for this segment,
        and whether the ROI was saturated during this segment's sweep.
    """
    phases = np.linspace(0, 2 * np.pi, NUM_PHASES, endpoint=False)
    r0, r1, c0, c1 = get_segment_indices(seg_idx, num_seg_x, num_seg_y,
                                            sq_y0, sq_y1, sq_x0, sq_x1)
    saturated = False

    # STEP 1: Pre-compute all bitmaps (vectorised, no DMD interaction)
    bitmaps = compute_phase_bitmaps(current_field, r0, r1, c0, c1, phases)

    # STEP 2: Overwrite the persistent DMD sequence with new bitmaps
    DMD.SeqPut(imgData=bitmaps, SequenceId=seqid)

    # STEP 2b: Set timing AFTER images are loaded (ALP API requires SeqPut first)
    picture_us, illum_us = dmd_times(exposure_us)
    DMD.SetTiming(
        SequenceId=seqid,
        illuminationTime=illum_us,
        pictureTime=picture_us,
        synchDelay=0,
        synchPulseWidth=DMD_SYNCH_PULSE_WIDTH,
        triggerInDelay=None,
    )

    # STEP 3: Start DMD playback (loop=False: plays once then stops)
    DMD.Run(loop=False, SequenceId=seqid)

    # STEP 4: Grab exactly NUM_PHASES frames (one per DMD frame trigger)
    intensities = []
    for i, phase in enumerate(phases):
        # RetrieveResult blocks until the DMD sync trigger fires
        grabResult = camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            frame = image.GetArray()
            grabResult.Release()

            intens = measure_roi(frame, roi_x, roi_y, roi_w, roi_h)
            intensities.append(intens)

            if intens > SATURATION_THRESHOLD:
                saturated = True

            # REAL-TIME DISPLAY
            display = frame.copy()
            cv2.rectangle(display, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            label = f"Seg {seg_idx} | Phase {np.degrees(phase):.0f} deg | Max: {intens:.0f}/255"
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if saturated:
                cv2.putText(display, "SATURATED - reduce exposure!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow('Basler Live View', display)
            cv2.waitKey(1)

        else:
            grabResult.Release()
            intensities.append(0.0)

    DMD.Halt()

    # STEP 5: Select best phase and apply to cumulative field
    best_idx = np.argmax(intensities)
    best_phase = phases[best_idx]
    best_intensity = max(intensities)
    print(f"  Segment {seg_idx} best phase = {np.degrees(best_phase):.1f} deg")

    current_field[r0:r1, c0:c1] *= np.exp(1j * best_phase)
    return best_phase, saturated, best_intensity


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------

def main():
    os.makedirs("results", exist_ok=True)
    # --- USER SETTINGS (set after debug) ---
    roi_x, roi_y = 916, 864
    roi_w, roi_h = 48, 44
    # --- end user settings ---

    # =========================================================
    # 1. DMD Initialisation
    # =========================================================
    DMD = ALP4(version='4.3', libDir='ALP-4.3')
    DMD.Initialize()
    DMD.DevControl(ALP_SYNCH_POLARITY, ALP_LEVEL_LOW)

    # =========================================================
    # 2. Initial Hologram Projection (single-frame for live view)
    # =========================================================
    patch_size = 256
    height, width = 1600, 2560

    # SQUARE
    side = 400
    half_side = side//2
    cy, cx = height // 2, width // 2

    square = np.zeros((height,width), dtype=np.float64)
    square[cy - half_side : cy + half_side, cx - half_side : cx + half_side] = 255
    working_bitmap = compute_lee_hologram(square)

    exposure_time = 50000  # 50ms exposure

    seqid = DMD.SeqAlloc(nbImg=1, bitDepth=8)
    DMD.ProjControl(ALP_PROJ_INVERSION, 1)
    DMD.SeqPut(imgData=working_bitmap.reshape(1, *working_bitmap.shape))
    init_p, init_i = dmd_times(exposure_time)
    DMD.SetTiming(SequenceId=seqid, illuminationTime=init_i, pictureTime=init_p,
                  synchDelay=0, synchPulseWidth=DMD_SYNCH_PULSE_WIDTH, triggerInDelay=None)
    DMD.Run(loop=True)

    # =========================================================
    # 3. Camera Initialisation
    # =========================================================
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTimeAbs.SetValue(exposure_time)
    camera.TriggerSource.SetValue("Line3")
    camera.TriggerMode.SetValue("Off")  # Start in free-running for live view
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # =========================================================
    # 4. Live View & Interactive Controls
    # =========================================================
    display_seqid = None  # Allocated during optimisation if needed
    cv2.namedWindow('Basler Live View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Basler Live View', 1024, 768)

    # --- ROI selection state (mouse callback) ---
    roi_selecting = False
    roi_drag_start = None
    roi_pending = None  # rectangle drawn but not yet confirmed

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_selecting, roi_drag_start, roi_pending, roi_x, roi_y, roi_w, roi_h
        if not roi_selecting:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_drag_start = (x, y)
            roi_pending = None
        elif event == cv2.EVENT_MOUSEMOVE and roi_drag_start is not None:
            x0 = min(roi_drag_start[0], x)
            y0 = min(roi_drag_start[1], y)
            x1 = max(roi_drag_start[0], x)
            y1 = max(roi_drag_start[1], y)
            roi_pending = (x0, y0, x1, y1)
        elif event == cv2.EVENT_LBUTTONUP and roi_drag_start is not None:
            x0 = min(roi_drag_start[0], x)
            y0 = min(roi_drag_start[1], y)
            x1 = max(roi_drag_start[0], x)
            y1 = max(roi_drag_start[1], y)
            if x1 - x0 > 2 and y1 - y0 > 2:
                roi_x, roi_y = x0, y0
                roi_w, roi_h = x1 - x0, y1 - y0
                print(f"ROI set: ({roi_x}, {roi_y}) size {roi_w}x{roi_h}")
            else:
                print("ROI too small, ignoring.")
            roi_drag_start = None
            roi_pending = None
            roi_selecting = False

    cv2.setMouseCallback('Basler Live View', mouse_callback)

    print("Press 'h' to load final_hologram.png onto the DMD.")
    print("Press 's' to save a screenshot.")
    print("Press 'r' to draw ROI (click-drag-release on the live view).")
    print("Press 'e' / 'w' to increase / decrease exposure time.")
    print("Press 'd' to locate the patch (prints brightest pixel).")
    print("Press 'm' to run the full optimisation (all segments).")
    print("Press 'q' to quit.")

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(100000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            frame = image.GetArray()
            raw_frame = frame.copy()
            grabResult.Release()

            if roi_x != 0 or roi_y != 0:
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0,255,0), 2)
            if roi_pending is not None:
                cv2.rectangle(frame, (roi_pending[0], roi_pending[1]),
                              (roi_pending[2], roi_pending[3]), (0, 255, 255), 2)
            if roi_selecting:
                cv2.putText(frame, "DRAW ROI (click-drag-release)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Exposure: {exposure_time} us", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow('Basler Live View', frame)

            key = cv2.waitKey(1) & 0xFF

            # --- ROI SELECTION MODE ('r' key) ---
            if key == ord('r'):
                roi_selecting = True
                roi_drag_start = None
                roi_pending = None
                print("ROI selection: click and drag on the live view, then release.")

            elif key == ord('q') or key == 27:
                break

            # --- SCREENSHOT ('s' key) ---
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"C:\\Users\\Alan_\\Downloads\\Summer Internship\\work\\DMD_code\\results\\screenshot_{timestamp}.png"
                cv2.imwrite(filename, raw_frame)
                print(f"Screenshot saved: {filename}")

            # --- LOAD FINAL HOLOGRAM ('h' key) ---
            elif key == ord('h'):
                if os.path.exists("final_hologram.png"):
                    holo = cv2.imread("final_hologram.png", cv2.IMREAD_GRAYSCALE)
                    if holo is not None and holo.shape == (height, width):
                        DMD.Halt()
                        DMD.SeqPut(imgData=holo.reshape(1, *holo.shape))
                        DMD.ProjControl(ALP_PROJ_INVERSION, 1)
                        h_p, h_i = dmd_times(exposure_time)
                        DMD.SetTiming(SequenceId=seqid, illuminationTime=h_i,
                                      pictureTime=h_p, synchDelay=0,
                                      synchPulseWidth=DMD_SYNCH_PULSE_WIDTH, triggerInDelay=None)
                        DMD.Run(loop=True)
                        print("final_hologram.png loaded and projected.")
                    else:
                        print("final_hologram.png has wrong dimensions or couldn't be read.")
                else:
                    print("final_hologram.png not found. Run optimisation ('m') first.")

            # Block debug/optimise keys while drawing ROI
            elif roi_selecting:
                continue

            # --- EXPOSURE UP ('e' key) ---
            elif key == ord('e'):
                exposure_time = min(exposure_time + 10000, 1000000)
                camera.ExposureTimeAbs.SetValue(exposure_time)
                picture_us, illum_us = dmd_times(exposure_time)
                DMD.SetTiming(SequenceId=seqid, illuminationTime=illum_us,
                              pictureTime=picture_us, synchDelay=0,
                              synchPulseWidth=DMD_SYNCH_PULSE_WIDTH, triggerInDelay=None)
                print(f"Exposure: {exposure_time} us | DMD picture time: {picture_us} us")

            # --- EXPOSURE DOWN ('w' key) ---
            elif key == ord('w'):
                exposure_time = max(exposure_time - 10000, 1000)
                camera.ExposureTimeAbs.SetValue(exposure_time)
                picture_us, illum_us = dmd_times(exposure_time)
                DMD.SetTiming(SequenceId=seqid, illuminationTime=illum_us,
                              pictureTime=picture_us, synchDelay=0,
                              synchPulseWidth=DMD_SYNCH_PULSE_WIDTH, triggerInDelay=None)
                print(f"Exposure: {exposure_time} us | DMD picture time: {picture_us} us")

            # --- DEBUG MODE ('d' key) ---
            elif key == ord('d'):
                patch_size = 256
                field = np.zeros((1600, 2560), dtype=complex)
                field[:patch_size, :patch_size] = 1.0
                debug_bitmap = compute_lee_hologram(field)

                DMD.Halt()
                DMD.SeqPut(imgData=debug_bitmap.reshape(1, *debug_bitmap.shape))
                DMD.ProjControl(ALP_PROJ_INVERSION, 1)
                dbg_p, dbg_i = dmd_times(exposure_time)
                DMD.SetTiming(SequenceId=seqid, pictureTime=dbg_p,
                              synchDelay=0, synchPulseWidth=DMD_SYNCH_PULSE_WIDTH, triggerInDelay=None)
                DMD.Run(loop=True)
                time.sleep(0.1)

                grabResult2 = camera.RetrieveResult(100000, pylon.TimeoutHandling_ThrowException)
                if grabResult2.GrabSucceeded():
                    img = converter.Convert(grabResult2)
                    frame2 = img.GetArray()
                    grabResult2.Release()
                    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
                    print(f"Brightest pixel at {max_loc} with intensity {max_val}")
                    cv2.circle(frame2, max_loc, 15, (0,0,255), 3)
                    cv2.imshow('Basler Live View', frame2)
                    cv2.waitKey(0)
                else:
                    grabResult2.Release()

                # Restore the working pattern after debug
                DMD.Halt()
                DMD.SeqPut(imgData=working_bitmap.reshape(1, *working_bitmap.shape))
                DMD.ProjControl(ALP_PROJ_INVERSION, 1)
                restore_p, restore_i = dmd_times(exposure_time)
                DMD.SetTiming(SequenceId=seqid, illuminationTime=restore_i, pictureTime=restore_p,
                              synchDelay=0, synchPulseWidth=DMD_SYNCH_PULSE_WIDTH, triggerInDelay=None)
                DMD.Run(loop=True)

            # --- FULL OPTIMISATION MODE ('m' key) ---
            elif key == ord('m'):
                if roi_x == 0 and roi_y == 0:
                    print("Please run debug ('d') first to locate the patch and set ROI coordinates.")
                    continue

                num_seg_x, num_seg_y = 12, 12
                total_seg = num_seg_x * num_seg_y

                # Square region bounds (must match the square defined in section 2)
                sq_y0, sq_y1 = cy - half_side, cy + half_side
                sq_x0, sq_x1 = cx - half_side, cx + half_side

                # Full DMD-sized field: amplitude=1 inside square, 0 outside
                current_field = np.zeros((1600, 2560), dtype=complex)
                current_field[sq_y0:sq_y1, sq_x0:sq_x1] = 1.0
                current_exposure_us = exposure_time

                # --- OPTIMISATION SETUP (done once, before both rounds) ---
                DMD.Halt()

                DMD.FreeSeq(seqid)
                seqid = DMD.SeqAlloc(nbImg=NUM_PHASES, bitDepth=8)
                display_seqid = DMD.SeqAlloc(nbImg=1, bitDepth=8)
                DMD.ProjControl(ALP_PROJ_INVERSION, 1)

                switch_camera_mode(camera, 'triggered')

                print(f"Starting sequential optimisation ({total_seg} segments, "
                      f"{NUM_PHASES} phases, {dmd_times(exposure_time)[0]/1000:.0f}ms frame time)...")

                for l in range(1):
                    print(f"\n{'='*50}")
                    print(f"[Round {l+1}/2] Starting...")
                    print(f"{'='*50}")
                    t_start = time.perf_counter()

                    best_intensities = []
                    best_phases = []
                    for seg_idx in range(total_seg):
                        print(f"\n[Round {l+1}/2] Optimizing segment {seg_idx+1}/{total_seg}")
                        best_phase, saturated, best_intensity = optimize_segment(
                            DMD, camera, converter, seqid,
                            seg_idx, roi_x, roi_y, roi_w, roi_h,
                            current_field, num_seg_x, num_seg_y,
                            sq_y0, sq_y1, sq_x0, sq_x1,
                            exposure_us=exposure_time)

                        best_intensities.append(best_intensity)
                        best_phases.append(best_phase)

                        current_exposure_us = adjust_exposure_for_saturation(
                            camera, current_exposure_us, saturated)

                        if PROGRESS_INTERVAL > 0 and (seg_idx + 1) % PROGRESS_INTERVAL == 0:
                            interim_bitmap = compute_lee_hologram(current_field)
                            switch_camera_mode(camera, 'free_running')
                            DMD.SeqPut(imgData=interim_bitmap.reshape(1, *interim_bitmap.shape), SequenceId=display_seqid)
                            DMD.ProjControl(ALP_PROJ_INVERSION, 1)
                            interim_p, interim_i = dmd_times(exposure_time)
                            DMD.SetTiming(SequenceId=display_seqid,
                                          illuminationTime=interim_i,
                                          pictureTime=interim_p,
                                          synchDelay=0, synchPulseWidth=DMD_SYNCH_PULSE_WIDTH,
                                          triggerInDelay=None)
                            DMD.Run(loop=True, SequenceId=display_seqid)
                            time.sleep(0.3)
                            DMD.Halt()
                            switch_camera_mode(camera, 'triggered')

                    elapsed = time.perf_counter() - t_start
                    print(f"\n[Round {l+1}/2] Complete in {elapsed:.1f}s "
                          f"({elapsed/total_seg:.2f}s per segment)")

                    # Save per-round results (no DMD or camera mode changes)
                    np.save(
                        "results/dynamic_inverted.npy",
                        current_field)
                    df = pd.DataFrame({'Phase': best_phases, 'Intensity': best_intensities})
                    df.to_csv(
                        "results/dynamic_inverted.csv",
                        index=False)
                    print(f"Round {l+1}/2 results saved to results/2loops/")

                # --- POST-OPTIMISATION TEARDOWN (after both rounds) ---
                switch_camera_mode(camera, 'free_running')
                DMD.FreeSeq(seqid)
                if display_seqid is not None:
                    DMD.FreeSeq(display_seqid)
                    display_seqid = None
                seqid = DMD.SeqAlloc(nbImg=1, bitDepth=8)

                # Project the final optimised hologram
                print("\nGenerating final hologram for projection...")
                final_bitmap = compute_lee_hologram(current_field)
                DMD.SeqPut(imgData=final_bitmap.reshape(1, *final_bitmap.shape), SequenceId=seqid)
                DMD.ProjControl(ALP_PROJ_INVERSION, 1)
                final_p, final_i = dmd_times(exposure_time)
                DMD.SetTiming(SequenceId=seqid, illuminationTime=final_i, pictureTime=final_p,
                              synchDelay=0, synchPulseWidth=DMD_SYNCH_PULSE_WIDTH,
                              triggerInDelay=None)
                DMD.Run(loop=True, SequenceId=seqid)
                print("Final hologram projected. ROI intensity should be maximised.")
        else:
            grabResult.Release()

    # =========================================================
    # 6. Cleanup
    # =========================================================
    DMD.Halt()
    DMD.FreeSeq(seqid)
    if display_seqid is not None:
        DMD.FreeSeq(display_seqid)
    DMD.Free()
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()