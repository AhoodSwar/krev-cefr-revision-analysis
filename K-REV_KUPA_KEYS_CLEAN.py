#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KUPA-KEYS Revision Features: end-to-end revision-feature extraction from
revision episodes identified automatically by K-REV Python Library and
ordinal CEFR modelling.

INPUTS
------
Task2.csv
DLLA.csv

PIPELINE
--------
raw Task2 events
 -> text-state/cursor reconstruction
 -> K-REV revision episodes
 -> Revision features extraction
 -> composite CEFR target
 -> final common population
 -> Model 
 -> 10-fold StratifiedKFold OOF evaluation
 -> OOF confusion-matrix counts
 -> full-sample standardized coefficients
 -> McKelvey-Zavoina pseudo-R2


Public data:
https://huggingface.co/datasets/ALTACambridge/KUPA-KEYS


"""

from pathlib import Path
from difflib import SequenceMatcher
import argparse
import json
import warnings
import numpy as np
import pandas as pd

from krev import KeystrokeText
from mord import LogisticIT
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, mean_absolute_error
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
GT1000_MS = 1000.0

# Non-printable keys excluded from text reconstruction.
NON_PRINTABLE_KEYS = {
    'Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'Tab', 'Escape',
    'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown',
    'PageUp', 'PageDown', 'Home', 'End', 'Insert',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
    'AltGraph', 'ContextMenu', 'Dead'
}
LEVELS = ["B1","B2","C1","C2"]
YMAP = {"B1":0,"B2":1,"C1":2,"C2":3}
NEW_GT1000 = "n_backspace_sequences_with_gt1000ms_intervals"

REVISION13 = [
    "n_revision_episodes",
    "total_revision_time_ratio",
    "revision_regularity_sec_sd",
    "ratio_backspace_seq_gt3_in_revisions",
    NEW_GT1000,
    "mean_time_to_edit_ms_in_revisions",
    "median_backspaces_per_revision",
    "median_chars_deleted_per_revision",
    "median_chars_inserted_per_revision",
    "median_time_from_start_process_sec",
    "median_pause_before_revision_sec",
    "n_backspacing_sequences_in_revisions",
    "mean_backspaces_per_sequence",
]

def process_single_participant(user_df):
    """
    Fully robust processor: Uses 'down' events as the primary source of truth,
    maintains strict 1:1 length parity, dynamically anchors mid-session 
    captures to eliminate intermediate validation text mismatches, and 
    extracts the exact timestamp of every keystroke.
    """
    user_df = user_df.copy()
    
    # Sort chronologically with deterministic tie-breaking: click < down < input/insert < up < capture
    user_df['type_rank'] = user_df['type'].map({
        'click': 0,
        'down': 1,
        'input': 2,
        'insert': 2,
        'up': 3,
        'capture': 4
    })

    user_df = user_df.sort_values(
        ['time', 'type_rank'],
        kind='mergesort'
    ).reset_index(drop=True)
    
    text = ""
    current_cursor = 0
    current_selection_end = 0
    
    # --- PROPERLY INITIALIZED LISTS ---
    text_states = []
    cursor_positions = []
    time_list = []
    
    for idx, row in user_df.iterrows():
        event_type = row['type']
        
        # 1. Track text-selection boundaries.
        if pd.notna(row['range_start']) and pd.notna(row['range_end']):
            start = min(max(0, int(row['range_start'])), len(text))
            end = min(max(0, int(row['range_end'])), len(text))

            if start > end:
                start, end = end, start

            current_cursor, current_selection_end = start, end

        # 2. Handle heavy insertions (Pastes / Autocorrect replacements)
        if event_type in ['input', 'insert']:
            inp_text = str(row['text']) if pd.notna(row['text']) else ""

            if len(inp_text) > 1:
                if current_cursor != current_selection_end:
                    text = (
                        text[:current_cursor]
                        + inp_text
                        + text[current_selection_end:]
                    )
                    current_cursor += len(inp_text)

                else:
                    text = (
                        text[:current_cursor]
                        + inp_text
                        + text[current_cursor:]
                    )
                    current_cursor += len(inp_text)

                current_selection_end = current_cursor

        # 3. # Use key-down events as the primary representation of physical keystrokes.
        elif event_type == 'down':
            key = str(row['key']) if pd.notna(row['key']) else ''

            is_ctrl = (
                pd.notna(row['ctrl_key'])
                and bool(row['ctrl_key'])
            )

            is_meta = (
                pd.notna(row['meta_key'])
                and bool(row['meta_key'])
            )
            
            if is_ctrl or is_meta:
                pass

            elif key == 'Backspace':
                if current_cursor == current_selection_end:
                    if current_cursor > 0:
                        text = (
                            text[:current_cursor - 1]
                            + text[current_cursor:]
                        )
                        current_cursor -= 1

                else:
                    del_start = min(
                        current_cursor,
                        current_selection_end
                    )

                    del_end = max(
                        current_cursor,
                        current_selection_end
                    )

                    text = (
                        text[:del_start]
                        + text[del_end:]
                    )

                    current_cursor = del_start

                current_selection_end = current_cursor
                
            elif key == 'Delete':
                if current_cursor == current_selection_end:
                    if current_cursor < len(text):
                        text = (
                            text[:current_cursor]
                            + text[current_cursor + 1:]
                        )

                else:
                    del_start = min(
                        current_cursor,
                        current_selection_end
                    )

                    del_end = max(
                        current_cursor,
                        current_selection_end
                    )

                    text = (
                        text[:del_start]
                        + text[del_end:]
                    )

                    current_cursor = del_start

                current_selection_end = current_cursor
                
            elif key == 'Enter':
                if current_cursor != current_selection_end:
                    del_start = min(
                        current_cursor,
                        current_selection_end
                    )

                    del_end = max(
                        current_cursor,
                        current_selection_end
                    )

                    text = (
                        text[:del_start]
                        + text[del_end:]
                    )

                    current_cursor = del_start

                text = (
                    text[:current_cursor]
                    + '\n'
                    + text[current_cursor:]
                )

                current_cursor += 1
                current_selection_end = current_cursor
                
            elif len(key) == 1 and key not in NON_PRINTABLE_KEYS:
                if current_cursor != current_selection_end:
                    del_start = min(
                        current_cursor,
                        current_selection_end
                    )

                    del_end = max(
                        current_cursor,
                        current_selection_end
                    )

                    text = (
                        text[:del_start]
                        + key
                        + text[del_end:]
                    )

                    current_cursor = del_start + 1

                else:
                    text = (
                        text[:current_cursor]
                        + key
                        + text[current_cursor:]
                    )

                    current_cursor += 1

                current_selection_end = current_cursor

        # 4. Dynamic capture synchronization
        elif event_type == 'capture':
            cap_text = row['text']

            if pd.notna(cap_text):
                text = str(cap_text)

                current_cursor = min(
                    current_cursor,
                    len(text)
                )

                current_selection_end = min(
                    current_selection_end,
                    len(text)
                )

        # Record state strictly for ALL down events
        if event_type == 'down':
            text_states.append(text)
            cursor_positions.append(current_cursor)
            time_list.append(row['time'])
            
    return text_states, cursor_positions, time_list

def safe_lookup(values, idx):
    try:
        i = int(idx)
        if 0 <= i < len(values):
            return values[i]
    except Exception:
        pass
    return np.nan

def finite_float(x):
    try:
        x=float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan

def event_rank(series):
    return series.map({"click":0,"down":1,"input":2,"insert":2,"up":3,"capture":4}).fillna(5)

def as_bool(v):
    if isinstance(v,str):
        return v.strip().lower() in {"true","1","yes"}
    return bool(v) if pd.notna(v) else False

def sync_provenance_after_capture(old_text, old_origins, new_text):
    # Synchronize character-origin timestamps with the reconstructed text state.
    old_text = "" if old_text is None else str(old_text)
    new_text = "" if new_text is None else str(new_text)
    m = SequenceMatcher(None, old_text, new_text, autojunk=False)
    new_origins = [np.nan] * len(new_text)
    for tag,i1,i2,j1,j2 in m.get_opcodes():
        if tag == "equal":
            new_origins[j1:j2] = old_origins[i1:i2]
    return new_origins

def build_task2_trace(user_df):
    """Accepted KUPA raw-trace logic: one row per sorted key-down event."""
    df=user_df.copy()
    df["_rank"]=event_rank(df["type"])
    df=df.sort_values(["time","_rank"],kind="mergesort").reset_index(drop=True)
    text=""; origin_times=[]; current_cursor=0; current_selection_end=0
    down_rows=[]; previous_down_time=np.nan; down_index=-1

    for _,row in df.iterrows():
        typ=row["type"]
        event_time=float(row["time"]) if pd.notna(row.get("time")) else np.nan

        if pd.notna(row.get("range_start")) and pd.notna(row.get("range_end")):
            start=min(max(0,int(row["range_start"])),len(text))
            end=min(max(0,int(row["range_end"])),len(text))
            if start>end: start,end=end,start
            current_cursor=start; current_selection_end=end

        if typ in {"input","insert"}:
            inp=str(row["text"]) if pd.notna(row.get("text")) else ""
            if len(inp)>1:
                lo=min(current_cursor,current_selection_end); hi=max(current_cursor,current_selection_end)
                text=text[:lo]+inp+text[hi:]
                origin_times=origin_times[:lo]+[event_time]*len(inp)+origin_times[hi:]
                current_cursor=lo+len(inp); current_selection_end=current_cursor

        elif typ=="down":
            down_index += 1
            key=str(row["key"]) if pd.notna(row.get("key")) else ""
            ctrl=as_bool(row.get("ctrl_key")); meta=as_bool(row.get("meta_key"))
            pause_before=(event_time-previous_down_time
                          if pd.notna(event_time) and pd.notna(previous_down_time) else np.nan)
            deleted_origin=np.nan; deleted_count=0

            if not (ctrl or meta):
                lo=min(current_cursor,current_selection_end); hi=max(current_cursor,current_selection_end)
                if key=="Backspace":
                    if lo==hi:
                        if current_cursor>0:
                            deleted_origin=origin_times[current_cursor-1] if current_cursor-1 < len(origin_times) else np.nan
                            deleted_count=1
                            text=text[:current_cursor-1]+text[current_cursor:]
                            if current_cursor-1 < len(origin_times): del origin_times[current_cursor-1]
                            current_cursor-=1
                    else:
                        # When multiple selected characters are deleted at once,
                        # no single deleted-origin time is assigned.
                        deleted_count=hi-lo
                        text=text[:lo]+text[hi:]; del origin_times[lo:hi]; current_cursor=lo
                    current_selection_end=current_cursor
                elif key=="Delete":
                    if lo==hi:
                        if current_cursor<len(text):
                            deleted_count=1
                            text=text[:current_cursor]+text[current_cursor+1:]
                            if current_cursor < len(origin_times): del origin_times[current_cursor]
                    else:
                        deleted_count=hi-lo
                        text=text[:lo]+text[hi:]; del origin_times[lo:hi]; current_cursor=lo
                    current_selection_end=current_cursor
                elif key=="Enter":
                    if lo!=hi:
                        text=text[:lo]+text[hi:]; del origin_times[lo:hi]
                    text=text[:lo]+"\n"+text[lo:]
                    origin_times=origin_times[:lo]+[event_time]+origin_times[lo:]
                    current_cursor=lo+1; current_selection_end=current_cursor
                elif len(key)==1 and key not in NON_PRINTABLE_KEYS:
                    if lo!=hi:
                        text=text[:lo]+text[hi:]; del origin_times[lo:hi]
                    text=text[:lo]+key+text[lo:]
                    origin_times=origin_times[:lo]+[event_time]+origin_times[lo:]
                    current_cursor=lo+1; current_selection_end=current_cursor

            down_rows.append({
                "down_index":down_index,"time_ms":event_time,"key":key,
                "pause_2_before_ms":pause_before,
                "deleted_origin_time_ms":deleted_origin,
                "deleted_char_count":deleted_count,
            })
            previous_down_time=event_time

        elif typ=="capture":
            cap=row.get("text")
            if pd.notna(cap):
                cap=str(cap)
                if cap!=text:
                    origin_times=sync_provenance_after_capture(text,origin_times,cap)
                text=cap
                current_cursor=min(current_cursor,len(text))
                current_selection_end=min(current_selection_end,len(text))

    trace=pd.DataFrame(down_rows)
    if trace.empty: return trace, pd.DataFrame()

    trace["is_backspace"]=trace["key"].eq("Backspace")
    trace["backspace_sequence_id"]=pd.Series(pd.NA,index=trace.index,dtype="Int64")
    seq_records=[]; seq_id=0; i=0
    while i<len(trace):
        if not bool(trace.loc[i,"is_backspace"]):
            i+=1; continue
        start=i
        while i+1<len(trace) and bool(trace.loc[i+1,"is_backspace"]):
            i+=1
        end=i; seq_id+=1
        trace.loc[start:end,"backspace_sequence_id"]=seq_id
        seq=trace.loc[start:end].copy()
        pauses=pd.to_numeric(seq["pause_2_before_ms"],errors="coerce")
        qualifies=bool((pauses>GT1000_MS).fillna(False).any())
        first_bs=seq.iloc[0]["time_ms"]; erased=seq.iloc[0]["deleted_origin_time_ms"]
        tte=(float(first_bs)-float(erased)
             if pd.notna(first_bs) and pd.notna(erased) and float(first_bs)>=float(erased)
             else np.nan)
        seq_records.append({
            "backspace_sequence_id":seq_id,
            "sequence_start_down_index":int(trace.loc[start,"down_index"]),
            "sequence_end_down_index":int(trace.loc[end,"down_index"]),
            "n_backspaces":int(len(seq)),
            "is_gt3":bool(len(seq)>3),
            "gt1000ms_qualifies":qualifies,
            "time_to_edit_ms":tte,
        })
        i+=1
    return trace,pd.DataFrame(seq_records)

def transition_edit_chunks(before,after):
    before="" if before is None else str(before); after="" if after is None else str(after)
    m=SequenceMatcher(None,before,after,autojunk=False)
    deleted=[]; inserted=[]
    for tag,i1,i2,j1,j2 in m.get_opcodes():
        if tag=="delete": deleted.append(before[i1:i2])
        elif tag=="insert": inserted.append(after[j1:j2])
        elif tag=="replace":
            deleted.append(before[i1:i2]); inserted.append(after[j1:j2])
    return [x for x in deleted if x],[x for x in inserted if x]

def episode_character_counts(text_list,s,e):
    try: s=int(s); e=int(e)
    except Exception: return np.nan,np.nan
    if s<0 or e<=s or e>=len(text_list): return np.nan,np.nan
    nd=ni=0
    for i in range(s+1,e+1):
        d,ins=transition_edit_chunks(text_list[i-1],text_list[i])
        nd += sum(len(x) for x in d); ni += sum(len(x) for x in ins)
    return float(nd),float(ni)

def union_duration_sec(starts,ends):
    pairs=[]
    for a,b in zip(starts,ends):
        if pd.notna(a) and pd.notna(b):
            a=float(a); b=float(b)
            if b<a: a,b=b,a
            pairs.append((a,b))
    if not pairs: return 0.0
    pairs.sort(); total=0.0; cs,ce=pairs[0]
    for s,e in pairs[1:]:
        if s<=ce: ce=max(ce,e)
        else: total += ce-cs; cs,ce=s,e
    total += ce-cs
    return total/1000.0

def make_pipeline():
    return Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler()),
        ("model",LogisticIT(alpha=1.0)),
    ])

def metric_dict(y_true,y_pred):
    yt=np.asarray(y_true,dtype=int); yp=np.asarray(y_pred,dtype=int)
    return {
        "accuracy":accuracy_score(yt,yp),
        "balanced_accuracy":balanced_accuracy_score(yt,yp),
        "mae":mean_absolute_error(yt,yp),
        "adjacent_accuracy":float(np.mean(np.abs(yt-yp)<=1)),
    }

def pseudo_r2(fitted,X):
    Xi=fitted.named_steps["imputer"].transform(X)
    Xs=fitted.named_steps["scaler"].transform(Xi)
    beta=np.asarray(fitted.named_steps["model"].coef_,dtype=float)
    latent=np.asarray(Xs).dot(beta)
    v=float(np.var(latent,ddof=0)); resid=float(np.pi**2/3)
    return v/(v+resid)


# Build the Backspace-sequence trace used for sequence-level measures (Pacquetet, 2024).
def safe_float(x):
    """Convert a scalar to float; return NaN for missing/non-numeric values."""
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def levenshtein_distance(a, b):
    """Return the Levenshtein edit distance between two strings."""
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (ca != cb)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def build_pacquetet_trace_and_sequences(user_df, pid):
    df = user_df.copy()
    df["_rank"] = event_rank(df["type"])
    df = df.sort_values(["time", "_rank"], kind="mergesort").reset_index(drop=True)

    text = ""
    origins = []
    current_cursor = 0
    current_selection_end = 0
    down_rows = []
    previous_down_time = np.nan
    down_index = -1

    for _, row in df.iterrows():
        typ = row["type"]
        event_time = safe_float(row.get("time"))

        if pd.notna(row.get("range_start")) and pd.notna(row.get("range_end")):
            start = min(max(0, int(row["range_start"])), len(text))
            end = min(max(0, int(row["range_end"])), len(text))
            if start > end:
                start, end = end, start
            current_cursor = start
            current_selection_end = end

        if typ in {"input", "insert"}:
            inp = str(row["text"]) if pd.notna(row.get("text")) else ""
            if len(inp) > 1:
                lo = min(current_cursor, current_selection_end)
                hi = max(current_cursor, current_selection_end)
                text = text[:lo] + inp + text[hi:]
                origins = origins[:lo] + [event_time] * len(inp) + origins[hi:]
                current_cursor = lo + len(inp)
                current_selection_end = current_cursor

        elif typ == "down":
            down_index += 1
            key = str(row["key"]) if pd.notna(row.get("key")) else ""
            ctrl = as_bool(row.get("ctrl_key"))
            meta = as_bool(row.get("meta_key"))

            deleted_text = ""
            deleted_origin_time_ms = np.nan
            produced_text = ""

            if not (ctrl or meta):
                if key == "Backspace":
                    lo = min(current_cursor, current_selection_end)
                    hi = max(current_cursor, current_selection_end)
                    if lo != hi:
                        deleted_text = text[lo:hi]
                        # For deletion of multiple selected characters, no single
                        # deleted-origin time is assigned.
                        text = text[:lo] + text[hi:]
                        del origins[lo:hi]
                        current_cursor = lo
                    elif current_cursor > 0:
                        deleted_text = text[current_cursor - 1:current_cursor]
                        deleted_origin_time_ms = origins[current_cursor - 1]
                        text = text[:current_cursor - 1] + text[current_cursor:]
                        del origins[current_cursor - 1]
                        current_cursor -= 1
                    current_selection_end = current_cursor

                elif key == "Delete":
                    lo = min(current_cursor, current_selection_end)
                    hi = max(current_cursor, current_selection_end)
                    if lo != hi:
                        text = text[:lo] + text[hi:]
                        del origins[lo:hi]
                        current_cursor = lo
                    elif current_cursor < len(text):
                        text = text[:current_cursor] + text[current_cursor + 1:]
                        del origins[current_cursor]
                    current_selection_end = current_cursor

                elif key == "Enter":
                    lo = min(current_cursor, current_selection_end)
                    hi = max(current_cursor, current_selection_end)
                    if lo != hi:
                        text = text[:lo] + text[hi:]
                        del origins[lo:hi]
                        current_cursor = lo
                    text = text[:current_cursor] + "\n" + text[current_cursor:]
                    origins.insert(current_cursor, event_time)
                    current_cursor += 1
                    current_selection_end = current_cursor
                    produced_text = "\n"

                elif len(key) == 1 and key not in NON_PRINTABLE_KEYS:
                    lo = min(current_cursor, current_selection_end)
                    hi = max(current_cursor, current_selection_end)
                    text = text[:lo] + key + text[hi:]
                    origins = origins[:lo] + [event_time] + origins[hi:]
                    current_cursor = lo + 1
                    current_selection_end = current_cursor
                    produced_text = key

            pause_before_ms = (
                event_time - previous_down_time
                if pd.notna(event_time) and pd.notna(previous_down_time)
                else np.nan
            )

            down_rows.append({
                "id": pid,
                "down_index": down_index,
                "time_ms": event_time,
                "key": key,
                "pause_before_ms": pause_before_ms,
                "deleted_text": deleted_text,
                "deleted_origin_time_ms": deleted_origin_time_ms,
                "produced_text": produced_text,
            })
            previous_down_time = event_time

        elif typ == "capture":
            cap = row.get("text")
            if pd.notna(cap):
                cap = str(cap)
                if cap != text:
                    origins = sync_provenance_after_capture(text, origins, cap)
                text = cap
                current_cursor = min(current_cursor, len(text))
                current_selection_end = min(current_selection_end, len(text))

    trace = pd.DataFrame(down_rows)
    if trace.empty:
        return trace, pd.DataFrame()

    seq_rows = []
    i = 0
    seq_id = 0

    while i < len(trace):
        if trace.loc[i, "key"] != "Backspace":
            i += 1
            continue

        start = i
        while i + 1 < len(trace) and trace.loc[i + 1, "key"] == "Backspace":
            i += 1
        end = i
        seq_id += 1

        seq = trace.loc[start:end].copy()
        n_bs = len(seq)

        # Reconstruct the deleted text in its original order.
        # Consecutive Backspaces delete characters from right to left.
        backspaced = ""
        for chunk in seq["deleted_text"].fillna(""):
            if chunk:
                backspaced = str(chunk) + backspaced

        first_bs_time = safe_float(seq.iloc[0]["time_ms"])
        first_deleted_origin = safe_float(seq.iloc[0]["deleted_origin_time_ms"])
        tte = np.nan
        tte_exact_observed = False
        if pd.notna(first_bs_time) and pd.notna(first_deleted_origin) and first_bs_time >= first_deleted_origin:
            tte = first_bs_time - first_deleted_origin
            tte_exact_observed = True

       # Identify the text typed after a Backspace sequence as the replacement.
       # A replacement is considered exact when the number of text-producing
       # key-downs matches the number of Backspaces before another Backspace occurs.        repl_chars = []
        ambiguous = False
        j = end + 1
        while j < len(trace) and len(repl_chars) < n_bs:
            keyj = trace.loc[j, "key"]
            if keyj == "Backspace":
                ambiguous = True
                break
            produced = str(trace.loc[j, "produced_text"] or "")
            if produced:
                repl_chars.extend(list(produced))
            j += 1

        replacement_complete = (len(repl_chars) >= n_bs) and not ambiguous
        replaced = "".join(repl_chars[:n_bs]) if replacement_complete else ""

        lev = np.nan
        rda = np.nan
        if replacement_complete and len(backspaced) > 0:
            lev = float(levenshtein_distance(backspaced, replaced))
            rda = lev - float(n_bs)

        seq_rows.append({
            "id": pid,
            "backspace_sequence_id": seq_id,
            "sequence_start_down_index": int(trace.loc[start, "down_index"]),
            "sequence_end_down_index": int(trace.loc[end, "down_index"]),
            "pacquetet_n_backspaces_in_sequence": int(n_bs),
            "pacquetet_backspaced_sequence": backspaced,
            "pacquetet_time_to_edit_ms": tte,
            "pacquetet_time_to_edit_exact_observed": tte_exact_observed,
            "pacquetet_replaced_sequence": replaced,
            "pacquetet_replacement_complete_unambiguous": replacement_complete,
            "pacquetet_levenshtein_backspaced_vs_replaced": lev,
            "pacquetet_relative_deletion_amount": rda,
        })

        i += 1

    return trace, pd.DataFrame(seq_rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task2",required=True)
    ap.add_argument("--dlla",required=True)
    ap.add_argument("--output-dir",default="KUPAKEYS_Revision13_reproduction")
    args=ap.parse_args()
    OUT=Path(args.output_dir); OUT.mkdir(parents=True,exist_ok=True)

    task2=pd.read_csv(args.task2,low_memory=False)
    dlla=pd.read_csv(args.dlla,low_memory=False)
    for df in (task2,dlla):
        if "id" not in df.columns: raise KeyError("Required column 'id' not found.")
        df["id"]=df["id"].astype(str).str.strip()

    # 1. Task-2 reconstruction, validation, and K-REV
    #
    # Reconstruct and validate Task-2 writings before K-REV analysis.
    # Only writings that pass reconstruction validation are used to
    # extract K-REV revision episodes.
    recon_all={}
    validation_rows=[]
    failed_validation_ids=[]

    for pid,g in task2.groupby("id",sort=False):
        text_list,cursor_list,time_list=process_single_participant(g)
        recon_all[pid]={"text_list":text_list,"cursor_list":cursor_list,"time_list":time_list}

        passed=True
        error=""
        try:
            user_df=g.sort_values("time")
            down_events=user_df[user_df["type"].eq("down")]

            # Structural validation check.
            if len(text_list) != len(down_events):
                raise AssertionError(
                    f"text_states length ({len(text_list)}) != down events ({len(down_events)})"
                )

            # Cursor-bounds validation check.
            for i,(txt,cur) in enumerate(zip(text_list,cursor_list)):
                if not (0 <= cur <= len(txt)):
                    raise AssertionError(
                        f"keystroke {i}: Cursor ({cur}) out of bounds for text length ({len(txt)})"
                    )

            # Final-capture validation and tolerances.
            capture_events=user_df[user_df["type"].eq("capture")]
            if not capture_events.empty and len(text_list)>0:
                actual_text=capture_events.iloc[-1]["text"]
                final_actual_text=str(actual_text) if pd.notna(actual_text) else ""
                final_simulated_text=text_list[-1]

                match_exact=(final_simulated_text == final_actual_text)
                match_tolerant=(
                    abs(len(final_simulated_text)-len(final_actual_text)) <= 2
                    and final_simulated_text.replace(" ","")
                        == final_actual_text.replace(" ","")
                )
                match_in_flight=(
                    (final_simulated_text.startswith(final_actual_text)
                     or final_actual_text.startswith(final_simulated_text))
                    and abs(len(final_simulated_text)-len(final_actual_text)) <= 3
                )
                if not (match_exact or match_tolerant or match_in_flight):
                    raise AssertionError(
                        "Final text mismatch: "
                        f"expected tail={final_actual_text[-40:]!r}; "
                        f"got tail={final_simulated_text[-40:]!r}"
                    )

        except AssertionError as exc:
            passed=False
            error=str(exc)
            failed_validation_ids.append(pid)

        validation_rows.append({
            "id":pid,
            "validation_passed":passed,
            "error":error,
            "n_text_states":len(text_list),
        })

    validation_df=pd.DataFrame(validation_rows)
    validation_df.to_csv(OUT/"00_Task2_reconstruction_validation.csv",index=False)

    # Extract K-REV revision episodes from writings that passed
    # Task-2 reconstruction validation.
    recon={pid:v for pid,v in recon_all.items() if pid not in set(failed_validation_ids)}

    print(f"Task-2 reconstruction validation: passed={len(recon)}, failed={len(failed_validation_ids)}")

    episode_rows=[]; summary=[]
    for pid,kd in recon.items():
        text_list=kd["text_list"]
        cursor_list=kd["cursor_list"]
        time_list=kd["time_list"]
        kt=KeystrokeText(text_list,cursor_list)
        revs=kt.extract_revisions()
        valid=[finite_float(x) for x in time_list if np.isfinite(finite_float(x))]
        ss=valid[0] if valid else np.nan; se=valid[-1] if valid else np.nan
        summary.append({"id":pid,"n_text_states":len(text_list),"n_revisions_krev":len(revs),
                        "session_start_time_ms":ss,"session_end_time_ms":se})
        for rid,rev in enumerate(revs,1):
            s=int(rev.index_start); e=int(rev.index_end)
            episode_rows.append({
                "id":pid,"revision_id":rid,"index_start":s,"index_end":e,
                "revision_start_time_ms":safe_lookup(time_list,s),
                "revision_end_time_ms":safe_lookup(time_list,e),
                "sequencing":getattr(rev,"sequencing",None),
            })
    episodes=pd.DataFrame(episode_rows)
    pd.DataFrame(summary).to_csv(OUT/"01_KREV_summary.csv",index=False)
    episodes.to_csv(OUT/"02_KREV_revision_episodes.csv",index=False)

    # 2. Raw traces and global physical Backspace sequences
    traces={}; seqs={}
    task2_groups={str(pid):g.copy() for pid,g in task2.groupby("id",sort=False)}
    for pid in recon:
        g=task2_groups.get(str(pid))
        if g is None:
            traces[pid]=pd.DataFrame(); seqs[pid]=pd.DataFrame()
        else:
            tr,sq=build_task2_trace(g)
            traces[pid]=tr; seqs[pid]=sq

    # Exact event-count/timestamp alignment audit.
    align=[]
    for pid,kd in recon.items():
        tr=traces.get(pid,pd.DataFrame())
        kt=np.asarray([finite_float(x) for x in kd["time_list"]],dtype=float)
        rt=(pd.to_numeric(tr["time_ms"],errors="coerce").to_numpy(float)
            if not tr.empty else np.asarray([],dtype=float))
        same=len(kt)==len(rt)
        exact=False
        if same:
            sf=np.array_equal(np.isfinite(kt),np.isfinite(rt))
            mask=np.isfinite(kt)&np.isfinite(rt)
            exact=bool(sf and np.allclose(kt[mask],rt[mask],rtol=0,atol=1e-9))
        align.append({"id":pid,"n_krev":len(kt),"n_raw_down":len(rt),
                      "same_n":same,"exact_timestamp_alignment":exact})
    pd.DataFrame(align).to_csv(OUT/"03_KREV_raw_alignment.csv",index=False)

    # 3. Episode-level measures (Conijn et al., 2022) and writing-level aggregation.
    conijn_episode=[]; writing=[]
    for pid,kd in recon.items():
        eps=episodes.loc[episodes["id"].eq(pid)].sort_values("revision_id").copy()
        tr=traces.get(pid,pd.DataFrame()); sq=seqs.get(pid,pd.DataFrame())
        tl=kd["time_list"]; texts=kd["text_list"]
        valid=[finite_float(x) for x in tl if np.isfinite(finite_float(x))]
        session_start=valid[0] if valid else np.nan
        session_end=valid[-1] if valid else np.nan
        aligned=(len(tr)==len(tl))
        if aligned and len(tr):
            a=np.asarray([finite_float(x) for x in tl])
            b=pd.to_numeric(tr["time_ms"],errors="coerce").to_numpy(float)
            mask=np.isfinite(a)&np.isfinite(b)
            aligned=bool(np.array_equal(np.isfinite(a),np.isfinite(b)) and
                         np.allclose(a[mask],b[mask],rtol=0,atol=1e-9))

        covered=set()
        crows=[]
        # Revision-level measures based on Conijn et al. (2022).
        # Each measure is calculated per K-REV revision episode and then
        # aggregated to the writing level using the median.
        for r in eps.itertuples(index=False):
            s=int(r.index_start); e=int(r.index_end); lo,hi=sorted((s,e))
            nd,ni=episode_character_counts(texts,s,e)
            nbs=np.nan
            if aligned and not tr.empty:
                nbs=int(tr.loc[tr["down_index"].between(lo,hi,inclusive="both") &
                               tr["key"].astype(str).str.lower().eq("backspace")].shape[0])
            pause=tfs=np.nan
            first_edit=s+1
            if first_edit<len(texts) and len(str(texts[first_edit]))<len(str(texts[s])) and aligned:
                tb=finite_float(tl[s]); tf=finite_float(tl[first_edit])
                if np.isfinite(tb) and np.isfinite(tf) and tf>=tb: pause=(tf-tb)/1000.0
                if np.isfinite(tf) and np.isfinite(session_start) and tf>=session_start:
                    tfs=(tf-session_start)/1000.0
            rec={"id":pid,"revision_id":int(r.revision_id),
                 "conijn_n_backspaces":nbs,
                 "conijn_n_characters_deleted":nd,
                 "conijn_n_characters_inserted":ni,
                 "conijn_time_from_start_process_sec":tfs,
                 "conijn_pause_before_revision_sec":pause}
            crows.append(rec); conijn_episode.append(rec)
            if not tr.empty:
                ids=tr.loc[tr["down_index"].between(lo,hi,inclusive="both") &
                           tr["backspace_sequence_id"].notna(),
                           "backspace_sequence_id"].astype(int).tolist()
                covered.update(ids)

        cg=pd.DataFrame(crows)
        def med(col):
            if cg.empty: return np.nan
            x=pd.to_numeric(cg[col],errors="coerce").dropna()
            return float(x.median()) if len(x) else np.nan

        # Link physical Backspace sequences to K-REV revision episodes.
        # A sequence is included when any Backspace down-index from that sequence
        # falls within a K-REV revision episode. This linkage is used for the
        # >3-Backspace proportion (Pacquetet, 2024), the study-specific >1,000-ms
        # interval count, and mean Time-to-Edit (Pacquetet, 2024).
        revision_backspace_seq=(sq.loc[sq["backspace_sequence_id"].isin(sorted(covered))]
                   .drop_duplicates("backspace_sequence_id")
                   if not sq.empty else pd.DataFrame())
        n_revision_backspace_seq=int(len(revision_backspace_seq))
        ratio_gt3=(float(revision_backspace_seq["is_gt3"].mean()) if n_revision_backspace_seq else np.nan)
        ngt1000=(int(revision_backspace_seq["gt1000ms_qualifies"].sum()) if n_revision_backspace_seq else 0)
        mean_tte=(float(pd.to_numeric(revision_backspace_seq["time_to_edit_ms"],errors="coerce").dropna().mean())
                  if n_revision_backspace_seq and pd.to_numeric(revision_backspace_seq["time_to_edit_ms"],errors="coerce").notna().any() else np.nan)

        # Backspace-sequence measures (Pacquetet, 2024).
        # A separate sequence trace is constructed to calculate the number of
        # Backspace sequences and the mean number of Backspaces per sequence.
        raw_user = task2.loc[task2["id"].astype(str).eq(pid)].copy()
        _, pacquetet_sq = build_pacquetet_trace_and_sequences(raw_user, pid)

        # Link Backspace sequences to K-REV revision episodes using the sequence
        # start index. Each physical sequence is counted once per writing, even
        # when it is associated with overlapping or nested K-REV episodes.
        revision_sequence_ids=set()
        if not pacquetet_sq.empty:
            for r in eps.itertuples(index=False):
                lo,hi=sorted((int(r.index_start),int(r.index_end)))
                ids=(pacquetet_sq.loc[
                    pacquetet_sq["sequence_start_down_index"].between(lo,hi,inclusive="both"),
                    "backspace_sequence_id"
                ].astype(int).tolist())
                revision_sequence_ids.update(ids)

        revision_sequence_table=(pacquetet_sq.loc[
            pacquetet_sq["backspace_sequence_id"].isin(sorted(revision_sequence_ids))
        ].drop_duplicates("backspace_sequence_id") if not pacquetet_sq.empty else pd.DataFrame())

        nseq=int(len(revision_sequence_table))
        mean_bs=(float(pd.to_numeric(
            revision_sequence_table["pacquetet_n_backspaces_in_sequence"], errors="coerce"
        ).mean()) if nseq else np.nan)

        nrev=int(len(eps))
        total_rev=union_duration_sec(eps["revision_start_time_ms"],eps["revision_end_time_ms"]) if nrev else 0.0
        elapsed=((session_end-session_start)/1000.0
                 if np.isfinite(session_start) and np.isfinite(session_end) and session_end>session_start else np.nan)
        ratio_time=total_rev/elapsed if pd.notna(elapsed) and elapsed>0 else np.nan
        starts=pd.to_numeric(eps["revision_start_time_ms"],errors="coerce").dropna().sort_values()
        if len(starts)>=2:
            intervals=starts.diff().dropna()/1000.0
            regularity=float(intervals.std(ddof=1)) if len(intervals)>=2 else np.nan
        else: regularity=np.nan

        writing.append({
            "id":pid,
            "n_revision_episodes":nrev,
            "total_revision_time_ratio":ratio_time,
            "revision_regularity_sec_sd":regularity,
            "ratio_backspace_seq_gt3_in_revisions":ratio_gt3,
            NEW_GT1000:ngt1000,
            "mean_time_to_edit_ms_in_revisions":mean_tte,
            "median_backspaces_per_revision":med("conijn_n_backspaces"),
            "median_chars_deleted_per_revision":med("conijn_n_characters_deleted"),
            "median_chars_inserted_per_revision":med("conijn_n_characters_inserted"),
            "median_time_from_start_process_sec":med("conijn_time_from_start_process_sec"),
            "median_pause_before_revision_sec":med("conijn_pause_before_revision_sec"),
            "n_backspacing_sequences_in_revisions":nseq,
            "mean_backspaces_per_sequence":mean_bs,
        })

    pd.DataFrame(conijn_episode).to_csv(OUT/"04_Conijn_episode_features.csv",index=False)
    feat_all=pd.DataFrame(writing)
    feat_all.to_csv(OUT/"05a_Revision13_all_validated_writings.csv",index=False)

    # Cross-task quality control:
    # Task-2 reconstruction validation retains 993 writings. Six additional
    # participants passed Task 2 but failed Task-1 validation; those six are
    # removed from Task 2 to preserve the matched cross-task sample.
    task1_only_failed_ids = {
        "R_7NXj5joxrx7p7rj",
        "R_27x1DrKYkonElql",
        "R_2SCyoayAejVMTNV",
        "R_PLD2rU5POOWtDTr",
        "R_2zvxRLmARd2daSv",
        "R_25Kn4eLb5d9NZEx",
    }
    found_task1_failures = sorted(set(feat_all["id"].astype(str)) & task1_only_failed_ids)
    pd.DataFrame({"id": found_task1_failures}).to_csv(OUT/"05b_cross_task_QC_exclusions.csv", index=False)
    print(f"Task-1-only validation failures found after Task-2 validation: {len(found_task1_failures)}")
    if len(found_task1_failures) != 6:
        raise RuntimeError(
            f"Cross-task QC mismatch: found {len(found_task1_failures)} of the 6 expected Task-1-only failures."
        )

    feat = feat_all.loc[~feat_all["id"].astype(str).isin(task1_only_failed_ids)].copy().reset_index(drop=True)
    feat.to_csv(OUT/"05_Revision13_writing_level_features.csv",index=False)
    print(f"After exact notebook cross-task QC: N={len(feat)}")
    if len(feat) != 987:
        raise RuntimeError(
            f"Cross-task-QC Revision13 population mismatch: N={len(feat)}; expected N=987."
        )

    # 4. Construct Composite CEFR target.
    marks=["mark_a0","mark_h1","mark_h2","mark_h3"]
    for c in marks:
        if c not in dlla.columns: raise KeyError(f"DLLA missing {c}")
        dlla[c]=pd.to_numeric(dlla[c],errors="coerce")
    dlla["cefr_composite_mean"]=dlla[marks].mean(axis=1)
    dlla["cefr_composite_rounded"]=np.round(dlla["cefr_composite_mean"])
    def to_cefr(v):
        if pd.isna(v): return np.nan
        s=int(v)
        if s==0:return np.nan
        if s in (1,2):return "A1"
        if s in (3,4):return "A2"
        if s in (5,6):return "B1"
        if s in (7,8):return "B2"
        if s in (9,10):return "C1"
        if s in (11,12,13):return "C2"
        return np.nan
    dlla["CEFR_target"]=dlla["cefr_composite_rounded"].apply(to_cefr)

    original=dlla["CEFR_target"].value_counts().reindex(["A1","A2","B1","B2","C1","C2"],fill_value=0)
    original.to_csv(OUT/"06_composite_CEFR_distribution_before_feature_merge.csv",header=["n"])

    data=feat.merge(dlla[["id","CEFR_target","cefr_composite_mean","cefr_composite_rounded"]],
                    on="id",how="inner",validate="one_to_one")
    data=data.loc[data["CEFR_target"].isin(LEVELS)].reset_index(drop=True)
    data["cefr_numeric"]=data["CEFR_target"].map(YMAP).astype(int)
    data.to_csv(OUT/"07_model_ready_common_population.csv",index=False)

    # Verify Task-2 reconstruction-validation counts:
    # 1,006 total, 993 passed, and 13 failed.
    if len(recon) != 993:
        raise RuntimeError(
            f"Validated Task-2 reconstruction population mismatch: N={len(recon)}; expected N=993."
        )

    exp_original={"A1":0,"A2":4,"B1":111,"B2":549,"C1":323,"C2":19}
    exp_model={"B1":110,"B2":540,"C1":314,"C2":19}
    obs_model=data["CEFR_target"].value_counts().reindex(LEVELS,fill_value=0).astype(int).to_dict()
    audit=pd.DataFrame({
        "expected_original":[exp_original[k] for k in ["A1","A2","B1","B2","C1","C2"]],
        "observed_original":[int(original[k]) for k in ["A1","A2","B1","B2","C1","C2"]],
    },index=["A1","A2","B1","B2","C1","C2"])
    audit.to_csv(OUT/"08_fidelity_audit.csv")
    if original.astype(int).to_dict()!=exp_original:
        raise RuntimeError("Composite CEFR target did not reproduce the audited original distribution.")
    if len(data)!=983 or obs_model!=exp_model:
        raise RuntimeError(f"Revision13 population mismatch: N={len(data)}, counts={obs_model}; expected N=983, {exp_model}")

    # 5. FINAL MODEL 4: 10-fold CV, OOF predictions.
    X=data[REVISION13].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    y=data["cefr_numeric"].astype(int)
    cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=RANDOM_STATE)
    fold_rows=[]; oof=[]
    for fold,(tri,tei) in enumerate(cv.split(X,y),1):
        m=make_pipeline(); m.fit(X.iloc[tri],y.iloc[tri])
        yp=np.asarray(m.predict(X.iloc[tei]),dtype=int)
        fold_rows.append({"fold":fold,**metric_dict(y.iloc[tei],yp)})
        z=data.iloc[tei][["id","CEFR_target"]].copy()
        z["fold"]=fold; z["true_numeric"]=y.iloc[tei].to_numpy(); z["pred_numeric"]=yp
        oof.append(z)
    folds=pd.DataFrame(fold_rows)
    folds.to_csv(OUT/"09_Model_10fold_CV_fold_results.csv",index=False)
    summary=pd.DataFrame({"metric":["accuracy","balanced_accuracy","mae","adjacent_accuracy"],
                          "mean":[folds[c].mean() for c in ["accuracy","balanced_accuracy","mae","adjacent_accuracy"]],
                          "sd":[folds[c].std(ddof=1) for c in ["accuracy","balanced_accuracy","mae","adjacent_accuracy"]]})
    summary.to_csv(OUT/"10_Model_10fold_CV_summary.csv",index=False)
    oo=pd.concat(oof,ignore_index=True); oo.to_csv(OUT/"11_Model4_OOF_predictions.csv",index=False)
    cm=confusion_matrix(oo["true_numeric"],oo["pred_numeric"],labels=[0,1,2,3])
    pd.DataFrame(cm,index=[f"true_{x}" for x in LEVELS],columns=[f"pred_{x}" for x in LEVELS]).to_csv(
        OUT/"12_Model_OOF_confusion_matrix_counts.csv")
    pd.DataFrame(classification_report(oo["true_numeric"],oo["pred_numeric"],
        labels=[0,1,2,3],target_names=LEVELS,zero_division=0,output_dict=True)).T.to_csv(
        OUT/"13_Model_OOF_classification_report.csv")

    final=make_pipeline(); final.fit(X,y)
    coef=pd.DataFrame({"feature":REVISION13,
                       "standardized_coefficient":final.named_steps["model"].coef_})
    coef.to_csv(OUT/"14_Model_full_sample_standardized_coefficients.csv",index=False)
    r2=pseudo_r2(final,X)
    pd.DataFrame([{"McKelvey_Zavoina_pseudo_R2":r2,"percent":100*r2}]).to_csv(
        OUT/"15_Model4_McKelvey_Zavoina_pseudo_R2.csv",index=False)

    print("\\nFINAL MODEL — 10-fold CV")
    print(summary.to_string(index=False))
    print("\\nPooled OOF:",metric_dict(oo["true_numeric"],oo["pred_numeric"]))
    print(f"McKelvey-Zavoina pseudo-R2 (full fitted model): {r2:.6f}")
    print("No holdout result is used as the primary reported evaluation.")
    print("No figures were generated.")

if __name__=="__main__":
    main()
