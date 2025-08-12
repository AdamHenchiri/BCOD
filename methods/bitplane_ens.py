import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from methods.base import TrackerMethod
from utils import extract_bit_plane_numba, compute_psi_numba, compute_weighted_histogram_optimized

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    ox = min(ax + aw, bx + bw)
    oy = min(ay + ah, by + bh)
    iw = max(0, ox - ix)
    ih = max(0, oy - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union == 0 else inter / union

class BitplaneEnsembleTracker(TrackerMethod):
    """
    Ensemble tracker: coarse-to-fine + psi + hist + ORB + gradient-hist.
    Normalisation adaptative sur la template pour scores en [0..1].
    """
    def __init__(self, init_frame, roi,
                 search_radius=60,
                 top_k=6,
                 pyr_scales=(0.6, 0.8, 1.0),
                 re_detect_interval=30,
                 update_alpha=0.08):
        x, y, w, h = map(int, roi)
        self.last_x, self.last_y = x, y
        self.w, self.h = w, h

        self.search_radius = search_radius
        self.top_k = top_k
        self.pyr_scales = pyr_scales
        self.re_detect_interval = re_detect_interval
        self.update_alpha = update_alpha  # running average for template update

        self.frame_count = 0
        self.confidence = 1.0

        # Template initial
        gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y:y + h, x:x + w].astype(np.uint8)

        # Precompute bitplanes/hist for template
        self.template_b6 = extract_bit_plane_numba(self.template, 6)
        self.template_b7 = extract_bit_plane_numba(self.template, 7)
        self.template_hist = compute_weighted_histogram_optimized(self.template)

        # gradient-oriented histogram for template
        self.template_grad_hist = self._grad_orientation_hist(self.template)

        # ORB features for template
        self.orb = cv2.ORB_create(nfeatures=300)
        self.kp_t, self.des_t = self.orb.detectAndCompute(self.template, None)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # calibration: scores template vs itself -> used to normalize component scores to [0..1]
        self._calibrate()

        # prepare reduced templates for matchTemplate on bitplane7
        self.template_b7_pyrs = {}
        for s in self.pyr_scales:
            if s == 1.0:
                t = (self.template_b7 * 255).astype(np.uint8)
            else:
                new_w = max(1, int(self.template_b7.shape[1] * s))
                new_h = max(1, int(self.template_b7.shape[0] * s))
                t = cv2.resize((self.template_b7 * 255).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.template_b7_pyrs[s] = t

    def _calibrate(self):
        """Compute self.baselines to normalize each score to ~[0..1]."""
        # psi on template vs itself
        I6 = extract_bit_plane_numba(self.template, 6)
        I7 = extract_bit_plane_numba(self.template, 7)
        psi_self = compute_psi_numba(I6, I7, I6, I7)  # likely max value
        self.psi_baseline = float(max(1.0, psi_self))

        # hist similarity of template vs itself
        hist_sim_self = cv2.compareHist(self.template_hist, self.template_hist, cv2.HISTCMP_CORREL)
        self.hist_baseline = float(max(1e-6, hist_sim_self))  # usually 1.0

        # gradient hist baseline
        gh = self.template_grad_hist
        self.grad_baseline = float(max(1e-6, np.sum(gh * gh)))  # used later as normalization helper

        # ORB baseline: match template to itself -> number of good matches
        if self.des_t is None or len(self.kp_t) < 2:
            self.orb_baseline = 1.0
        else:
            matches = self.bf.match(self.des_t, self.des_t)
            self.orb_baseline = float(max(1, len(matches)))

    def _grad_orientation_hist(self, img):
        """Return normalized histogram of gradient orientations (8 bins)."""
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # bin into 8 bins [0..360)
        bins = np.int32((ang // 45) % 8)
        hist = np.bincount(bins.ravel(), weights=mag.ravel(), minlength=8).astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist

    def _get_search_window(self, frame_gray):
        H, W = frame_gray.shape
        r = int(self.search_radius * (2.0 - self.confidence))
        x0 = max(0, self.last_x - r)
        x1 = min(W, self.last_x + r + self.w)
        y0 = max(0, self.last_y - r)
        y1 = min(H, self.last_y + r + self.h)
        return x0, x1, y0, y1

    def _coarse_match(self, frame_gray, global_search=False):
        x0, x1, y0, y1 = self._get_search_window(frame_gray) if not global_search else (0, frame_gray.shape[1], 0, frame_gray.shape[0])
        search_patch = frame_gray[y0:y1, x0:x1]
        candidates = []
        if search_patch.size == 0:
            return candidates

        for s in self.pyr_scales:
            if s == 1.0:
                sp = search_patch
            else:
                new_w = max(1, int(search_patch.shape[1] * s))
                new_h = max(1, int(search_patch.shape[0] * s))
                sp = cv2.resize(search_patch, (new_w, new_h), interpolation=cv2.INTER_AREA)

            sp_b7 = (extract_bit_plane_numba(sp, 7) * 255).astype(np.uint8)
            tpl = self.template_b7_pyrs[s]
            if tpl.shape[0] > sp_b7.shape[0] or tpl.shape[1] > sp_b7.shape[1]:
                continue
            res = cv2.matchTemplate(sp_b7, tpl, cv2.TM_CCOEFF_NORMED)
            flat = res.ravel()
            k = min(self.top_k, flat.size)
            if k == 0:
                continue
            idxs = np.argpartition(-flat, k - 1)[:k]
            ys, xs = np.unravel_index(idxs, res.shape)
            for xx_rel, yy_rel in zip(xs, ys):
                score_tm = float(res[yy_rel, xx_rel])
                if s == 1.0:
                    xx_full = x0 + xx_rel
                    yy_full = y0 + yy_rel
                else:
                    xx_full = x0 + int(xx_rel / s)
                    yy_full = y0 + int(yy_rel / s)
                candidates.append((xx_full, yy_full, s, score_tm))
        # sort and NMS-like dedupe
        candidates = sorted(candidates, key=lambda c: c[3], reverse=True)
        out = []
        seen = []
        min_dist = max(4, int(min(self.w, self.h) * 0.25))
        for c in candidates:
            cx, cy = c[0], c[1]
            if any((abs(cx - sx) < min_dist and abs(cy - sy) < min_dist) for sx, sy in seen):
                continue
            out.append(c)
            seen.append((cx, cy))
            if len(out) >= self.top_k:
                break
        return out

    def _eval_candidate(self, gray, cand, angles=(0, -8, 8)):
        xx, yy, s, tm_score = cand
        H, W = gray.shape
        x = max(0, min(W - self.w, int(xx)))
        y = max(0, min(H - self.h, int(yy)))
        roi = gray[y:y + self.h, x:x + self.w]
        if roi.shape[0] != self.h or roi.shape[1] != self.w:
            return None

        # component 1: psi (bitplanes 6+7)
        I6 = extract_bit_plane_numba(roi, 6)
        I7 = extract_bit_plane_numba(roi, 7)
        psi_val = float(compute_psi_numba(self.template_b6, self.template_b7, I6, I7))
        psi_score = min(1.0, psi_val / self.psi_baseline)

        # component 2: histogram similarity (cv2 correlate)
        roi_hist = compute_weighted_histogram_optimized(roi)
        hist_sim = float(cv2.compareHist(self.template_hist, roi_hist, cv2.HISTCMP_CORREL))
        # hist_sim typically near 1 for very similar; normalize by baseline
        hist_score = (hist_sim / self.hist_baseline + 1.0) / 2.0  # put roughly into [0..1]

        # component 3: gradient-orientation histogram correlation
        roi_grad = self._grad_orientation_hist(roi)
        # measure dot product or correlation
        grad_score = float(np.dot(self.template_grad_hist, roi_grad))  # in [0..1] roughly

        # component 4: ORB matching ratio
        kp_r, des_r = self.orb.detectAndCompute(roi, None)
        if des_r is None or self.des_t is None:
            orb_score = 0.0
        else:
            matches = self.bf.match(self.des_t, des_r)
            # ratio of matches to baseline (template self-match)
            orb_score = min(1.0, len(matches) / max(1.0, self.orb_baseline))

        # test small rotations and keep best
        best_comb = None
        for a in angles:
            if a != 0:
                M = cv2.getRotationMatrix2D((self.w // 2, self.h // 2), a, 1.0)
                roi_r = cv2.warpAffine(roi, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                # recompute components cheaply: hist + psi
                I6r = extract_bit_plane_numba(roi_r, 6)
                I7r = extract_bit_plane_numba(roi_r, 7)
                psi_val_r = float(compute_psi_numba(self.template_b6, self.template_b7, I6r, I7r))
                psi_score_r = min(1.0, psi_val_r / self.psi_baseline)
                roi_hist_r = compute_weighted_histogram_optimized(roi_r)
                hist_sim_r = float(cv2.compareHist(self.template_hist, roi_hist_r, cv2.HISTCMP_CORREL))
                hist_score_r = (hist_sim_r / self.hist_baseline + 1.0) / 2.0
                grad_score_r = float(np.dot(self.template_grad_hist, self._grad_orientation_hist(roi_r)))
                # combine preliminary (we keep ORB from unrotated only to save cost)
                comb_r = (0.45 * psi_score_r) + (0.25 * hist_score_r) + (0.15 * grad_score_r) + (0.15 * orb_score)
                if best_comb is None or comb_r > best_comb[0]:
                    best_comb = (comb_r, psi_score_r, hist_score_r, grad_score_r, orb_score, a)
            else:
                comb0 = (0.45 * psi_score) + (0.25 * hist_score) + (0.15 * grad_score) + (0.15 * orb_score)
                if best_comb is None or comb0 > best_comb[0]:
                    best_comb = (comb0, psi_score, hist_score, grad_score, orb_score, 0)

        if best_comb is None:
            return None

        score_final = best_comb[0]
        return (score_final, (x, y, self.w, self.h), best_comb[5])

    def update(self, frame):
        """
        Retourne (bbox, score)
        """
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Decide if global re-detect (every N frames or when confidence low)
        global_search = (self.frame_count % self.re_detect_interval == 0) or (self.confidence < 0.25)

        candidates = self._coarse_match(gray, global_search=global_search)
        if not candidates:
            # fallback: small local grid (coarse) to avoid losing target entirely
            xb, yb = max(0, self.last_x - 8), max(0, self.last_y - 8)
            return (xb, yb, self.w, self.h), 0.0

        best_score = -1.0
        best_bbox = (self.last_x, self.last_y, self.w, self.h)

        with ThreadPoolExecutor(max_workers=min(4, len(candidates))) as ex:
            futures = {ex.submit(self._eval_candidate, gray, c): c for c in candidates}
            for fut in as_completed(futures):
                res = fut.result()
                if res is None:
                    continue
                score_local, bbox_local, angle = res
                if score_local is None:
                    continue
                if score_local > best_score:
                    best_score = score_local
                    best_bbox = bbox_local

        # Update template conservatively if high confidence
        if best_score >= 0.75:
            x, y, w, h = best_bbox
            roi = gray[y:y + h, x:x + w]
            if roi.shape[0] == h and roi.shape[1] == w:
                # running average (uint8)
                self.template = cv2.addWeighted(self.template.astype(np.float32), 1.0 - self.update_alpha,
                                                roi.astype(np.float32), self.update_alpha, 0).astype(np.uint8)
                # recompute dependent structures (lightweight)
                self.template_b6 = extract_bit_plane_numba(self.template, 6)
                self.template_b7 = extract_bit_plane_numba(self.template, 7)
                self.template_hist = compute_weighted_histogram_optimized(self.template)
                self.template_grad_hist = self._grad_orientation_hist(self.template)
                self.kp_t, self.des_t = self.orb.detectAndCompute(self.template, None)
                self._calibrate()

        # update last position and confidence
        self.last_x, self.last_y = best_bbox[0], best_bbox[1]
        # map best_score to confidence dynamics
        if best_score >= 0.6:
            self.confidence = min(1.0, self.confidence + 0.12)
        else:
            self.confidence = max(0.05, self.confidence - 0.18)

        return best_bbox, float(best_score)
