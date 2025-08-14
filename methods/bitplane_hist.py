from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from methods.base import TrackerMethod
from utils import extract_bit_plane_numba, compute_psi_numba, compute_weighted_histogram_optimized


class BitplaneHistTracker(TrackerMethod):
    def __init__(self, init_frame, roi, score_threshold=0.5, search_radius=50, top_k=6, pyr_scales=(0.85, 0.9, 1.0, 1.1, 1.15)):
        # roi: x, y, w, h (float or int)
        x, y, w, h = map(int, roi)
        self.last_x, self.last_y = x, y
        self.w, self.h = w, h

        self.score_threshold = score_threshold  # score relatif [0..1] attendu (on combine et normalise)
        self.search_radius = search_radius
        self.top_k = top_k
        self.pyr_scales = pyr_scales
        self.confidence = 1.0

        gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y:y + h, x:x + w]

        # Pré-calc bitplanes & histogramme pour le template
        self.template_b6 = extract_bit_plane_numba(self.template, 6)
        self.template_b7 = extract_bit_plane_numba(self.template, 7)
        self.template_hist = compute_weighted_histogram_optimized(self.template)

        # Prepare template for each scale for matchTemplate (bitplane7)
        self.template_b7_pyrs = {}
        for s in self.pyr_scales:
            if s == 1.0:
                t = self.template_b7
            else:
                new_shape = (max(1, int(self.template_b7.shape[1] * s)), max(1, int(self.template_b7.shape[0] * s)))
                t = cv2.resize(self.template_b7.astype(np.uint8) * 255, new_shape, interpolation=cv2.INTER_AREA)
            # matchTemplate needs single channel uint8
            self.template_b7_pyrs[s] = (t.astype(np.uint8))

        self.last_angle = 0
    def _get_search_window(self, frame_gray):
        H, W = frame_gray.shape
        r = int(self.search_radius * (2.0 - self.confidence))
        x0 = max(0, self.last_x - r)
        x1 = min(W, self.last_x + r + self.w)
        y0 = max(0, self.last_y - r)
        y1 = min(H, self.last_y + r + self.h)
        return x0, x1, y0, y1

    def _coarse_match(self, frame_gray):
        """
        Using matchTemplate on bitplane7 (pyramid scales)
        Return a list of candidats (xx, yy, scale, score_tmplt) by desc ord.
        """
        x0, x1, y0, y1 = self._get_search_window(frame_gray)
        search_patch = frame_gray[y0:y1, x0:x1]
        candidates = []

        if search_patch.shape[0] < 1 or search_patch.shape[1] < 1:
            return candidates

        for s in self.pyr_scales:
            # resize search_patch and template to same scale
            if s == 1.0:
                sp = search_patch
            else:
                new_w = max(1, int(search_patch.shape[1] * s))
                new_h = max(1, int(search_patch.shape[0] * s))
                sp = cv2.resize(search_patch, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # compute bitplane7 of sp and convert to uint8 for matchTemplate
            sp_b7 = (extract_bit_plane_numba(sp, 7) * 255).astype(np.uint8)

            tpl = self.template_b7_pyrs[s]
            # if template larger than search skip this scale
            if tpl.shape[0] > sp_b7.shape[0] or tpl.shape[1] > sp_b7.shape[1]:
                continue

            # matchTemplate (gives score in [-1,1])
            res = cv2.matchTemplate(sp_b7, tpl, cv2.TM_CCOEFF_NORMED)
            # get top N peaks from res
            flat = res.ravel()
            k = min(self.top_k, flat.size)
            if k == 0:
                continue
            idxs = np.argpartition(-flat, k - 1)[:k]  # indices of top k (unsorted)
            # convert flat idx to 2D coords
            ys, xs = np.unravel_index(idxs, res.shape)
            for xx_rel, yy_rel in zip(xs, ys):
                score = float(res[yy_rel, xx_rel])
                if s == 1.0:
                    xx_full = x0 + xx_rel
                    yy_full = y0 + yy_rel
                    scale_used = 1.0
                else:
                    # coordinates were in scaled search patch: map to original
                    xx_full = x0 + int(xx_rel / s)
                    yy_full = y0 + int(yy_rel / s)
                    scale_used = s
                candidates.append((xx_full, yy_full, scale_used, score))

        # sort by score desc and deduplicate near positions
        candidates = sorted(candidates, key=lambda c: c[3], reverse=True)
        filtered = []
        seen = []
        min_dist = max(4, int(min(self.w, self.h) * 0.2))
        for c in candidates:
            cx, cy = c[0], c[1]
            if any((abs(cx - sx) < min_dist and abs(cy - sy) < min_dist) for sx, sy in seen):
                continue
            filtered.append(c)
            seen.append((cx, cy))
            if len(filtered) >= self.top_k:
                break
        return filtered

    def _eval_candidate(self, frame_gray, cand, angles=(0, 10, -10)):
        """
        Evaluate candidat :
         - extract ROI (w,h) around
         - compute psi (zeros) and histogram similarity
         - test littlr rotations (via cv2.warpAffine)
        """
        xx, yy, scale_used, tm_score = cand
        H, W = frame_gray.shape

        # ensure bbox within image
        x = max(0, min(W - self.w, int(xx)))
        y = max(0, min(H - self.h, int(yy)))

        best_local = None  # (score, (x,y,w,h), angle)
        roi = frame_gray[y:y + self.h, x:x + self.w]
        if roi.shape[0] != self.h or roi.shape[1] != self.w:
            return None

        # precompute histogram for ROI (unrotated first)
        roi_hist = compute_weighted_histogram_optimized(roi)
        hist_sim = cv2.compareHist(self.template_hist, roi_hist, cv2.HISTCMP_CORREL)
        # compute bitplanes for psi
        I6 = extract_bit_plane_numba(roi, 6)
        I7 = extract_bit_plane_numba(roi, 7)
        zeros = compute_psi_numba(self.template_b6, self.template_b7, I6, I7)

        # combine scores : normalized-ish (weights chosen empirically)
        base_score = 0.7 * float(zeros) + 0.3 * float(hist_sim)
        best_local = (base_score, (x, y, self.w, self.h), 0)

        # test small rotations (affine rotate about center)
        cx, cy = self.w // 2, self.h // 2
        for a in angles:
            if a == 0:
                continue
            M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
            rot = cv2.warpAffine(roi, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            roi_hist_r = compute_weighted_histogram_optimized(rot)
            hist_sim_r = cv2.compareHist(self.template_hist, roi_hist_r, cv2.HISTCMP_CORREL)
            I6r = extract_bit_plane_numba(rot, 6)
            I7r = extract_bit_plane_numba(rot, 7)
            zeros_r = compute_psi_numba(self.template_b6, self.template_b7, I6r, I7r)
            score_r = 0.7 * float(zeros_r) + 0.3 * float(hist_sim_r)
            if score_r > best_local[0]:
                best_local = (score_r, (x, y, self.w, self.h), a)

        return best_local  # or None

    def update(self, frame):
        """
        Retourne (bbox, score)
        bbox = (x,y,w,h) or None si non trouvé
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # Coarse stage: get top candidates via matchTemplate on bitplane7 pyramid
        candidates = self._coarse_match(gray)
        if not candidates:
            # fallback : small local grid around last position (as before, but reduced)
            fallback_bbox = (self.last_x, self.last_y, self.w, self.h)
            self.last_angle = 0
            return fallback_bbox, 0.0

        # Fine evaluation in parallel (candidates x small rotations)
        best_score = -np.inf
        best_bbox = (self.last_x, self.last_y, self.w, self.h)

        # limit the number of workers to avoid oversub
        max_workers = min(len(candidates), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
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
                    self.last_angle = angle

        # update last pos + confidence
        self.last_x, self.last_y = best_bbox[0], best_bbox[1]
        # psi retourne probablement un int ou float >0 ; ici on suppose best_score est dans un intervalle raisonnable
        if best_score >= self.score_threshold:
            self.confidence = min(1.0, self.confidence + 0.08)
        else:
            self.confidence = max(0.05, self.confidence - 0.15)

        return best_bbox, float(best_score if np.isfinite(best_score) else 0.0)