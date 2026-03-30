import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import tempfile
import os
import time
from collections import Counter
from scipy.stats import entropy as scipy_entropy

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Gait Recognition System",
    page_icon="walking",
    layout="wide"
)

st.title("Gait Recognition System")
st.markdown("**Open-Set Detection - Entropy Rejection - Weighted Voting - Sequence Quality Filtering**")
st.caption("Running with original 7 features -- compatible with existing model & norm files.")

# ==========================================
# PATHS
# ==========================================
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "gait_lstm_videos1_baseline.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_videos1_baseline.pkl")
NORM_PATH    = os.path.join(MODEL_DIR, "norm_videos1_baseline.pkl")

# ==========================================
# GLOBAL CONFIG
# ==========================================
SEQ_LEN              = 60
STRIDE               = 6
MAX_PROCESS_TIME_SEC = 40

# Global rejection thresholds -- used when no per-identity override exists
ENTROPY_REJECT_THRESHOLD = 0.60   # normalized entropy: above this -> Unknown
MIN_MARGIN               = 0.18   # best_conf - second_conf
MIN_AGREEMENT            = 0.50   # weighted vote share of winner
MIN_BEST_CONF            = 0.55   # absolute softmax confidence floor

# Sequence quality filter bounds
# A sequence is dropped if mean per-feature std is outside [SEQ_STD_MIN, SEQ_STD_MAX]
SEQ_STD_MIN = 0.5    # degrees -- below this: person is nearly stationary
SEQ_STD_MAX = 60.0   # degrees -- above this: pose is too noisy / occluded

# ==========================================
# PER-IDENTITY OVERRIDES
# ==========================================
# Keys must exactly match label encoder class names.
#
# max_entropy : per-person entropy ceiling (overrides ENTROPY_REJECT_THRESHOLD).
#               Relaxing this for a person is safe because intruders are still
#               caught by the margin gate (their margin is typically ~0.01-0.03).
#
# min_conf    : minimum softmax confidence for this person.
# min_margin  : minimum best-minus-second gap for this person.
# min_agreement: minimum weighted vote share for this person.
#
# Any key not present falls back to the global slider value.
PER_IDENTITY_THRESHOLDS = {
    "Om":    {"min_conf": 0.44, "min_margin": 0.28, "min_agreement": 0.45, "max_entropy": 0.72},
    "Varad": {"min_conf": 0.50, "min_margin": 0.18, "min_agreement": 0.50, "max_entropy": 0.65},
    # Add more entries as needed, e.g.:
    # "Alice": {"min_conf": 0.55, "min_margin": 0.20, "min_agreement": 0.50, "max_entropy": 0.60},
}

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(NORM_PATH, "rb") as f:
        mean, std = pickle.load(f)
    return model, le, mean, std


# ==========================================
# FEATURE HELPERS
# ==========================================
def angle(a, b, c):
    """Joint angle at vertex b, in degrees (0-180)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = (np.arctan2(c[1] - b[1], c[0] - b[0])
           - np.arctan2(a[1] - b[1], a[0] - b[0]))
    deg = abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg


def torso_tilt(ls, rs, lh, rh):
    """Angle of shoulder-to-hip vector from vertical, in degrees."""
    shoulder = (np.array(ls) + np.array(rs)) / 2
    hip      = (np.array(lh) + np.array(rh)) / 2
    v        = shoulder - hip
    vertical = np.array([0, -1])
    cos_val  = np.dot(v, vertical) / (np.linalg.norm(v) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_val, -1, 1)))


# ==========================================
# SEQUENCE QUALITY FILTER
# ==========================================
def is_good_sequence(seq, std_min, std_max):
    """
    Return True if the sequence contains genuine gait motion.
    Drops sequences that are nearly static or extremely noisy.
    """
    arr      = np.array(seq)
    mean_std = arr.std(axis=0).mean()
    return std_min < mean_std < std_max


# ==========================================
# WEIGHTED VOTING
# ==========================================
def weighted_vote(preds):
    """
    Weight each sequence's vote by its softmax confidence.
    Returns (winning_class_index, agreement_ratio).
    """
    vote_weights = {}
    for pred in preds:
        cls = int(np.argmax(pred))
        vote_weights[cls] = vote_weights.get(cls, 0.0) + float(pred[cls])

    winner    = max(vote_weights, key=vote_weights.get)
    total_w   = sum(vote_weights.values())
    agreement = vote_weights[winner] / total_w if total_w > 0 else 0.0
    return winner, agreement


# ==========================================
# OPEN-SET DECISION ENGINE
# ==========================================
def make_decision(avg_probs, preds, le,
                  slider_entropy, slider_conf, slider_margin, slider_agreement):
    """
    Four cascaded rejection gates. Returns (final_name, reject_reason, debug_dict).

    Gate order:
      1. Entropy   -- flat distribution signals unknown person
      2. Confidence -- top score too low
      3. Margin    -- top-2 too close  (primary intruder signal: ~0.01-0.03 for unknowns)
      4. Agreement -- weighted vote too split

    Per-identity overrides from PER_IDENTITY_THRESHOLDS take priority over slider values.
    The entropy gate uses the per-identity max_entropy so that naturally high-entropy
    walkers (like Om) are not rejected, while intruders still fail at the margin gate.
    """
    debug = {}

    # --- Ranking ---
    sorted_idx  = np.argsort(avg_probs)
    best_idx    = int(sorted_idx[-1])
    second_idx  = int(sorted_idx[-2])
    best_conf   = float(avg_probs[best_idx])
    second_conf = float(avg_probs[second_idx])
    margin      = best_conf - second_conf
    best_name   = le.inverse_transform([best_idx])[0]

    # --- Entropy ---
    raw_entropy        = float(scipy_entropy(avg_probs))
    max_entropy        = float(np.log(len(avg_probs)))
    normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 1.0

    # --- Per-identity threshold lookup (applied to ALL gates including entropy) ---
    ov         = PER_IDENTITY_THRESHOLDS.get(best_name, {})
    max_ent    = ov.get("max_entropy",   slider_entropy)
    min_conf   = ov.get("min_conf",      slider_conf)
    min_margin = ov.get("min_margin",    slider_margin)
    min_agr    = ov.get("min_agreement", slider_agreement)

    # --- Weighted vote ---
    voted_class, agreement = weighted_vote(preds)
    voted_name = le.inverse_transform([voted_class])[0]

    # --- Populate debug dict ---
    debug["best_name"]          = best_name
    debug["voted_name"]         = voted_name
    debug["best_conf"]          = best_conf
    debug["second_conf"]        = second_conf
    debug["margin"]             = margin
    debug["normalized_entropy"] = normalized_entropy
    debug["agreement"]          = agreement
    debug["max_ent_used"]       = max_ent
    debug["min_conf_used"]      = min_conf
    debug["min_margin_used"]    = min_margin
    debug["min_agr_used"]       = min_agr

    # --- Gate 1: Entropy (per-identity ceiling) ---
    if normalized_entropy > max_ent:
        debug["reject_reason"] = "Entropy too high ({:.3f} > {:.3f})".format(
            normalized_entropy, max_ent)
        return "Unknown", debug["reject_reason"], debug

    # --- Gate 2: Confidence ---
    if best_conf < min_conf:
        debug["reject_reason"] = "Confidence too low ({:.3f} < {:.3f})".format(
            best_conf, min_conf)
        return "Unknown", debug["reject_reason"], debug

    # --- Gate 3: Margin (strongest intruder discriminator) ---
    if margin < min_margin:
        debug["reject_reason"] = "Margin too small ({:.3f} < {:.3f})".format(
            margin, min_margin)
        return "Unknown", debug["reject_reason"], debug

    # --- Gate 4: Agreement ---
    if agreement < min_agr:
        debug["reject_reason"] = "Vote agreement too low ({:.3f} < {:.3f})".format(
            agreement, min_agr)
        return "Unknown", debug["reject_reason"], debug

    debug["reject_reason"] = None
    return voted_name, "Accepted", debug


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload walking video",
        type=["mp4", "avi", "mov", "MOV"]
    )

    st.divider()
    st.subheader("Threshold Tuning")
    st.caption("Per-identity overrides in PER_IDENTITY_THRESHOLDS take priority over these sliders.")

    entropy_thresh = st.slider(
        "Max entropy (global fallback)",
        0.30, 0.90, ENTROPY_REJECT_THRESHOLD, 0.01,
        help="Normalized Shannon entropy. Higher = flatter distribution = likely unknown."
    )
    min_conf_slider = st.slider(
        "Min confidence (global fallback)",
        0.30, 0.95, MIN_BEST_CONF, 0.01
    )
    min_margin_slider = st.slider(
        "Min margin -- best minus 2nd best (global fallback)",
        0.05, 0.50, MIN_MARGIN, 0.01
    )
    min_agreement_slider = st.slider(
        "Min vote agreement (global fallback)",
        0.30, 0.90, MIN_AGREEMENT, 0.01
    )

    st.divider()
    st.caption("Sequence quality filter")
    std_min = st.number_input("Min feature std (degrees)", 0.0, 5.0,  SEQ_STD_MIN, 0.05)
    std_max = st.number_input("Max feature std (degrees)", 5.0, 120.0, SEQ_STD_MAX, 1.0)


# ==========================================
# MAIN LOGIC
# ==========================================
if uploaded_file is not None:
    model, le, mean, std = load_resources()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded_file.read())
    tmp.flush()

    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap    = cv2.VideoCapture(tmp.name)
    frames = []

    st.divider()
    status_box   = st.info("Extracting gait features...")
    progress_bar = st.progress(0.0)
    start_time   = time.time()

    # ===============================
    # FEATURE EXTRACTION
    # ===============================
    while cap.isOpened():
        elapsed = time.time() - start_time
        if elapsed >= MAX_PROCESS_TIME_SEC:
            break
        progress_bar.progress(min(elapsed / MAX_PROCESS_TIME_SEC, 1.0))

        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            continue

        lm = results.pose_landmarks.landmark
        P  = mp_pose.PoseLandmark

        def p(i):
            return [lm[i].x, lm[i].y]

        try:
            rs = p(P.RIGHT_SHOULDER); ls = p(P.LEFT_SHOULDER)
            rh = p(P.RIGHT_HIP);      lh = p(P.LEFT_HIP)
            rk = p(P.RIGHT_KNEE);     lk = p(P.LEFT_KNEE)
            ra = p(P.RIGHT_ANKLE);    la = p(P.LEFT_ANKLE)
            rf = p(P.RIGHT_FOOT_INDEX); lf = p(P.LEFT_FOOT_INDEX)

            features = [
                angle(rh, rk, ra),           # right knee flexion
                angle(lh, lk, la),           # left  knee flexion
                angle(rs, rh, rk),           # right hip flexion
                angle(ls, lh, lk),           # left  hip flexion
                angle(rk, ra, rf),           # right ankle dorsiflexion
                angle(lk, la, lf),           # left  ankle dorsiflexion
                torso_tilt(ls, rs, lh, rh),  # trunk lean
            ]
            frames.append(features)
        except Exception:
            continue

    cap.release()
    status_box.success("Extraction complete -- {} frames captured".format(len(frames)))
    progress_bar.empty()

    # ===============================
    # BUILD & FILTER SEQUENCES
    # ===============================
    raw_sequences  = []
    good_sequences = []
    bad_count      = 0

    for i in range(0, len(frames) - SEQ_LEN, STRIDE):
        seq = frames[i: i + SEQ_LEN]
        raw_sequences.append(seq)
        if is_good_sequence(seq, std_min, std_max):
            good_sequences.append(seq)
        else:
            bad_count += 1

    st.subheader("Final Result")

    if len(good_sequences) == 0:
        st.error("Not enough quality gait sequences. Try a longer or clearer walking clip.")
    else:
        X         = (np.array(good_sequences, dtype=np.float32) - mean) / std
        preds     = model.predict(X, verbose=0)
        avg_probs = np.mean(preds, axis=0)

        # Run decision engine -- slider values used as global fallback
        final_name, reject_reason, debug = make_decision(
            avg_probs, preds, le,
            slider_entropy   = entropy_thresh,
            slider_conf      = min_conf_slider,
            slider_margin    = min_margin_slider,
            slider_agreement = min_agreement_slider,
        )

        # Unpack debug values
        best_conf        = debug["best_conf"]
        second_conf      = debug["second_conf"]
        margin           = debug["margin"]
        norm_ent         = debug["normalized_entropy"]
        agreement        = debug["agreement"]
        best_name        = debug["best_name"]
        voted_name       = debug["voted_name"]
        max_ent_used     = debug["max_ent_used"]
        min_conf_used    = debug["min_conf_used"]
        min_margin_used  = debug["min_margin_used"]
        min_agr_used     = debug["min_agr_used"]
        identity_override = best_name in PER_IDENTITY_THRESHOLDS

        total_time = time.time() - start_time

        # ── Result cards ──────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            if final_name != "Unknown":
                st.success("### Identity: {}".format(final_name))
            else:
                st.error("### Unknown / Intruder")
                if reject_reason:
                    st.caption("Rejected: {}".format(reject_reason))
                st.caption("Closest match: {}".format(best_name))

        with col2:
            st.metric("Confidence",      "{:.2f}%".format(best_conf * 100))
            st.metric("Entropy (norm.)", "{:.3f}".format(norm_ent))

        with col3:
            st.metric("Time",                 "{:.1f}s".format(total_time))
            st.metric("Quality seqs / Total", "{} / {}".format(
                len(good_sequences), len(raw_sequences)))

        # ── Debug panel ───────────────────────────────────────────────────────
        with st.expander("Debug Metrics", expanded=True):
            colA, colB, colC, colD = st.columns(4)

            with colA:
                st.metric("Best confidence", "{:.2f}%".format(best_conf * 100))
                st.metric("Second best",     "{:.2f}%".format(second_conf * 100))

            with colB:
                st.metric("Margin",         "{:.4f}".format(margin))
                st.metric("Vote agreement", "{:.4f}".format(agreement))

            with colC:
                st.metric("Norm. entropy", "{:.4f}".format(norm_ent))
                st.metric("Entropy gate",
                          "Pass" if norm_ent <= max_ent_used else "Fail")

            with colD:
                st.metric("Sequences used",    len(good_sequences))
                st.metric("Sequences dropped", bad_count)

            if identity_override:
                st.info("Per-identity overrides active for: **{}**".format(best_name))
            else:
                st.caption("Using global slider thresholds (no per-identity override).")

            st.markdown("**Decision gate status**")
            override_tag = "  [override]" if identity_override else ""
            gates = {
                "Entropy gate": (
                    "Pass" if norm_ent   <= max_ent_used   else "Fail",
                    "{:.3f} <= {:.3f}{}".format(norm_ent,   max_ent_used,   override_tag)
                ),
                "Confidence gate": (
                    "Pass" if best_conf  >= min_conf_used  else "Fail",
                    "{:.3f} >= {:.3f}{}".format(best_conf,  min_conf_used,  override_tag)
                ),
                "Margin gate": (
                    "Pass" if margin     >= min_margin_used else "Fail",
                    "{:.3f} >= {:.3f}{}".format(margin,     min_margin_used, override_tag)
                ),
                "Agreement gate": (
                    "Pass" if agreement  >= min_agr_used   else "Fail",
                    "{:.3f} >= {:.3f}{}".format(agreement,  min_agr_used,   override_tag)
                ),
            }
            st.table({k: list(v) for k, v in gates.items()})

        # ── Class probabilities ───────────────────────────────────────────────
        with st.expander("Class Probabilities"):
            prob_dict = {
                le.classes_[i]: "{:.2f}%".format(float(avg_probs[i]) * 100)
                for i in np.argsort(avg_probs)[::-1]
            }
            st.table(prob_dict)

        # ── Weighted voting breakdown ─────────────────────────────────────────
        with st.expander("Weighted Voting Breakdown"):
            vote_weights_display = {}
            for pred in preds:
                cls  = int(np.argmax(pred))
                name = le.inverse_transform([cls])[0]
                vote_weights_display[name] = (
                    vote_weights_display.get(name, 0.0) + float(pred[cls])
                )
            total_w = sum(vote_weights_display.values())
            vote_table = {
                k: "{:.3f} ({:.1f}%)".format(v, v / total_w * 100)
                for k, v in sorted(
                    vote_weights_display.items(), key=lambda x: x[1], reverse=True
                )
            }
            st.table(vote_table)

        # ── Features used ─────────────────────────────────────────────────────
        with st.expander("Features Used"):
            st.markdown(
                "| # | Feature | Description |\n"
                "|---|---------|-------------|\n"
                "| 1 | Right knee flexion       | Hip-Knee-Ankle angle |\n"
                "| 2 | Left knee flexion        | Hip-Knee-Ankle angle |\n"
                "| 3 | Right hip flexion        | Shoulder-Hip-Knee angle |\n"
                "| 4 | Left hip flexion         | Shoulder-Hip-Knee angle |\n"
                "| 5 | Right ankle dorsiflexion | Knee-Ankle-Foot angle |\n"
                "| 6 | Left ankle dorsiflexion  | Knee-Ankle-Foot angle |\n"
                "| 7 | Torso tilt               | Trunk lean from vertical |\n"
            )

else:
    st.info("Upload a walking video to begin.")
    with st.expander("About the decision pipeline"):
        st.markdown("""
        This system uses **4 cascaded rejection gates** to distinguish known individuals from unknowns.

        **Gate 1 -- Entropy:** If the softmax distribution is too flat, reject.
        Per-identity `max_entropy` overrides let naturally high-entropy walkers pass
        while intruders are caught at the margin gate below.

        **Gate 2 -- Confidence:** The top class probability must exceed a minimum floor.

        **Gate 3 -- Margin:** The gap between rank-1 and rank-2 must be large enough.
        This is the strongest intruder signal: untrained subjects score ~0.01-0.03
        because the model spreads probability nearly equally across all classes.

        **Gate 4 -- Agreement:** The weighted vote across all sequences must converge.

        **Per-identity overrides** in `PER_IDENTITY_THRESHOLDS` take priority over the
        sidebar sliders for all four gates. The debug panel shows which thresholds were
        actually used and whether an override was active for the detected person.

        **Sequence quality filter** drops sequences with too little motion or too much noise.

        **Weighted voting** gives more influence to high-confidence sequences.

        **7 original gait features** -- fully compatible with existing model and norm files.
        """)