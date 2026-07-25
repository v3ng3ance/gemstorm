#!/usr/bin/env python3
"""
ChordVision Pro — Acorduri per bătaie, pe măsuri, sincronizat cu piesa
Ca Chordify: fiecare beat are acordul lui, highlight în timp real, playhead animat.

pip install customtkinter librosa yt-dlp numpy scipy matplotlib Pillow pygame
"""
import os, sys, threading, tempfile, math, time
from pathlib import Path
from collections import Counter
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
except ImportError:
    print("Instalează: pip install customtkinter"); sys.exit(1)

try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()
    HAS_PYGAME = True
except Exception as e:
    HAS_PYGAME = False

# ─── Culori ───────────────────────────────────────────────────────────────────
BG     = '#0d0d12'; SURF   = '#16161f'; SURF2  = '#1e1e2a'; SURF3  = '#252535'
BORDER = '#2d2d42'; BORDER2= '#3a3a55'; TEXT   = '#eeeeff'; TEXT2  = '#8888bb'
TEXT3  = '#44446a'; ACCENT = '#7c6af7'; ACCENT2= '#a695ff'; GREEN  = '#4ade80'
YELLOW = '#fbbf24'; RED    = '#f87171'; CYAN   = '#38bdf8';  ORANGE = '#fb923c'
WHITE  = '#ffffff'
QCOL = {'major':'#7c6af7','minor':'#38bdf8','dom7':'#4ade80','minor7':'#22d3ee',
        'major7':'#a78bfa','dim':'#f87171','aug':'#fb923c','sus':'#fbbf24'}

# ─── Teoria muzicii ───────────────────────────────────────────────────────────
NOTE   = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
NOTEF  = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
KS_MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
MAJ_SC = [0,2,4,5,7,9,11]; MIN_SC = [0,2,3,5,7,8,10]
COF    = [0,7,2,9,4,11,6,1,8,3,10,5]

# suffix -> (intervale, ponderi, calitate afisata)
# Ponderile mai mici pe notele "de extensie" (7, b5 etc.) + penalizarea de mai jos
# fac ca un acord simplu sa nu mai piarda niciodata in fata unei variante cu mai
# multe note doar pentru ca acea nota suplimentara are putina energie reziduala.
CTMPL = {
    '':     ([0,4,7],      [1.0,1.0,1.0],      'major'),
    'm':    ([0,3,7],      [1.0,1.0,1.0],      'minor'),
    'dim':  ([0,3,6],      [1.0,1.0,1.0],      'dim'),
    'aug':  ([0,4,8],      [1.0,1.0,1.0],      'aug'),
    'sus2': ([0,2,7],      [1.0,0.9,1.0],      'sus'),
    'sus4': ([0,5,7],      [1.0,0.9,1.0],      'sus'),
    '7':    ([0,4,7,10],   [1.0,0.9,0.9,0.8],  'dom7'),
    'm7':   ([0,3,7,10],   [1.0,0.9,0.9,0.8],  'minor7'),
    'maj7': ([0,4,7,11],   [1.0,0.9,0.9,0.8],  'major7'),
    'm7b5': ([0,3,6,10],   [1.0,0.9,0.9,0.8],  'm7b5'),
    'dim7': ([0,3,6,9],    [1.0,0.9,0.9,0.9],  'dim7'),
}
_ALL_LABELS = [(r,suf) for r in range(12) for suf in CTMPL]
_TEMPLATE_CACHE = {}
def _template_vec(root,suf):
    key=(root,suf)
    if key in _TEMPLATE_CACHE: return _TEMPLATE_CACHE[key]
    ivs,wts,_ = CTMPL[suf]
    v=np.zeros(12)
    for iv,w in zip(ivs,wts): v[(root+iv)%12]=w
    _TEMPLATE_CACHE[key]=v
    return v

# Penalizare mica per nota "in plus" fata de un triad simplu (Occam's razor):
# un acord de 7 castiga doar cand a 7-a e clar prezenta in semnal, nu la egalitate.
COMPLEXITY_PENALTY = 0.03
# Sub acest nivel de energie bruta (inainte de normalizare) consideram bataia "fara acord"
NO_CHORD_ENERGY = 0.06

def pearson(a,b):
    a,b=np.array(a,float),np.array(b,float); ma,mb=a.mean(),b.mean()
    aa,bb=a-ma,b-mb; d=np.sqrt((aa**2).sum()*(bb**2).sum())
    return (aa*bb).sum()/d if d>1e-9 else 0.0

def detect_key(ch):
    best,bk,bm=-np.inf,0,'major'
    for k in range(12):
        s=pearson(ch,np.roll(KS_MAJ,-k))
        if s>best: best,bk,bm=s,k,'major'
        s=pearson(ch,np.roll(KS_MIN,-k))
        if s>best: best,bk,bm=s,k,'minor'
    return bk,bm

def _cosine(a,b):
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na<1e-9 or nb<1e-9: return 0.0
    return float(np.dot(a,b)/(na*nb))

def chord_scores(chroma):
    """Scor de similaritate cosinus intre chroma si fiecare acord candidat,
    cu o mica penalizare de complexitate. Inlocuieste vechea suma bruta care
    favoriza mereu acordurile cu mai multe note (bug principal de imprecizie)."""
    out=np.empty(len(_ALL_LABELS))
    for i,(r,suf) in enumerate(_ALL_LABELS):
        tmpl=_template_vec(r,suf)
        out[i]=_cosine(chroma,tmpl) - COMPLEXITY_PENALTY*(len(CTMPL[suf][0])-3)
    return out

def match_chord(chroma, raw_energy=None):
    if raw_energy is not None and raw_energy < NO_CHORD_ENERGY:
        return {'root':None,'suffix':'N.C.','name':'N.C.','quality':'none','root_pc':None}
    scores=chord_scores(chroma)
    i=int(np.argmax(scores))
    br,bs=_ALL_LABELS[i]
    return {'root':NOTE[br],'suffix':bs,'name':NOTE[br]+bs,
            'quality':CTMPL[bs][2],'root_pc':br}

def deg_color(root_pc, key_root, key_mode):
    if root_pc is None: return TEXT2
    sc=MAJ_SC if key_mode=='major' else MIN_SC
    for i,iv in enumerate(sc):
        if (key_root+iv)%12==root_pc:
            return [ACCENT,CYAN,'#60a5fa',GREEN,YELLOW,CYAN,RED][min(i,6)]
    return TEXT2

def _tint(hex_col, alpha, bg=(13,13,18)):
    try:
        r,g,b=int(hex_col[1:3],16),int(hex_col[3:5],16),int(hex_col[5:7],16)
        return '#{:02x}{:02x}{:02x}'.format(
            int(bg[0]+(r-bg[0])*alpha), int(bg[1]+(g-bg[1])*alpha), int(bg[2]+(b-bg[2])*alpha))
    except: return SURF2

def _smooth_scores(scores_seq, kernel=(0.15,0.7,0.15)):
    """Netezire temporala usoara pe scoruri (nu pe etichete) inainte de a alege
    acordul final per bataie -- reduce 'tremuratul' de la o bataie la alta fara
    sa strice tranzitiile reale de acord (care dureaza de obicei mai multe batai)."""
    arr=np.array(scores_seq); n=len(arr); k=len(kernel)//2
    out=np.zeros_like(arr)
    for t in range(n):
        acc=np.zeros(arr.shape[1]); wsum=0.0
        for j,w in enumerate(kernel):
            idx=t+j-k
            if 0<=idx<n: acc+=arr[idx]*w; wsum+=w
        out[t]=acc/wsum
    return out

# ─── Analiză audio – per bătaie ───────────────────────────────────────────────
def analyze(y, sr, cb=None):
    import librosa
    def p(v,m):
        if cb: cb(v,m)

    p(0.05, 'Separare armonică / percutivă...')
    y_harm, y_perc = librosa.effects.hpss(y)

    p(0.15, 'Beat tracking...')
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr,
                                                      units='frames', trim=False)
    tempo = float(np.atleast_1d(tempo_arr)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    p(0.25, f'{len(beat_times)} bătăi detectate la {tempo:.0f} BPM')

    p(0.30, 'Chroma CQT per bătaie...')
    hop    = 512
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop,
                                          bins_per_octave=36, norm=2)
    beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    # shape: (12, n_beats)

    p(0.38, 'Notă de bas per bătaie (registru grav)...')
    # Chroma calculat DOAR pe registrul grav (C1-C3-ish) ca sa gasim nota reala
    # canata de bas, nu doar radacina teoretica a acordului (care poate difera:
    # acorduri rasturnate / bas care se plimba pe alte note ale acordului).
    bass_chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop,
                                              fmin=librosa.note_to_hz('C1'),
                                              n_octaves=3, bins_per_octave=36, norm=2)
    beat_bass = librosa.util.sync(bass_chroma, beat_frames, aggregate=np.median)

    p(0.50, 'Scor acorduri per bătaie (similaritate cosinus)...')
    n_beats = beat_chroma.shape[1]
    norm_chroma = []
    raw_energy  = []
    for i in range(n_beats):
        c = beat_chroma[:, i].copy()
        mx = c.max()
        raw_energy.append(float(mx))
        norm_chroma.append(c/mx if mx>1e-9 else c)

    scores_seq = [chord_scores(c) for c in norm_chroma]
    p(0.60, 'Netezire temporală a scorurilor...')
    smoothed = _smooth_scores(scores_seq) if n_beats>=3 else np.array(scores_seq)

    raw=[]
    for i in range(n_beats):
        if raw_energy[i] < NO_CHORD_ENERGY:
            m = {'root':None,'suffix':'N.C.','name':'N.C.','quality':'none','root_pc':None}
        else:
            bi = int(np.argmax(smoothed[i]))
            br,bs = _ALL_LABELS[bi]
            m = {'root':NOTE[br],'suffix':bs,'name':NOTE[br]+bs,
                 'quality':CTMPL[bs][2],'root_pc':br}
        # nota reala de bas pentru aceasta bataie (independent de acordul de sus)
        bcol = beat_bass[:, i]
        bmax = float(bcol.max())
        if bmax > 1e-9 and m['root_pc'] is not None:
            bass_pc = int(np.argmax(bcol))
            m['bass_pc']  = bass_pc
            m['bass_note'] = NOTE[bass_pc]
            m['name_full'] = m['name'] if bass_pc==m['root_pc'] else f"{m['name']}/{NOTE[bass_pc]}"
        else:
            m['bass_pc']=m.get('root_pc'); m['bass_note']=m.get('root'); m['name_full']=m['name']
        t   = float(beat_times[i]) if i < len(beat_times) else i*60/tempo
        dur = float(beat_times[i+1]-beat_times[i]) if i+1<len(beat_times) else 60/tempo
        raw.append({'beat_idx':i, 'time':t, 'duration':dur, **m})

    p(0.72, 'Detecție metru...')
    beat_str = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop)
    bsv = librosa.util.sync(beat_str[np.newaxis,:], beat_frames, aggregate=np.max)[0]
    sig = _time_sig(bsv)
    p(0.76, f'Metru detectat: {sig}/4')

    for i, b in enumerate(raw):
        b['beat_in_measure'] = i % sig
        b['measure']         = i // sig
        b['beats_per_measure'] = sig

    p(0.82, 'Consolidare acorduri pe măsuri...')
    raw = _smooth(raw, sig)

    p(0.90, 'Tonalitate globală...')
    gc = beat_chroma.mean(axis=1); gc /= gc.max()+1e-9
    key_root, key_mode = detect_key(gc)

    for b in raw:
        col       = deg_color(b['root_pc'], key_root, key_mode)
        b['color'] = col
        b['fill']  = _tint(col, 0.18)

    p(1.0, 'Analiză completă!')
    return {
        'beats':      raw,
        'key_root':   key_root,
        'key_mode':   key_mode,
        'bpm':        round(tempo),
        'time_sig':   sig,
        'duration':   len(y)/sr,
        'n_measures': (raw[-1]['measure']+1) if raw else 0,
        'y': y, 'sr': sr,
    }

def _time_sig(strengths):
    if len(strengths) < 8: return 4
    ac = np.correlate(strengths, strengths, mode='full')[len(strengths)-1:]
    s3 = ac[3] if len(ac)>3 else 0
    s4 = ac[4] if len(ac)>4 else 0
    return 3 if s3 > s4*1.1 else 4

def _smooth(beats, sig):
    by_m = {}
    for b in beats: by_m.setdefault(b['measure'],[]).append(b)
    out = []
    for m in sorted(by_m):
        bs = by_m[m]; names=[b['name'] for b in bs]
        cnt=Counter(names); top,tn=cnt.most_common(1)[0]
        if tn/len(names) >= 0.55:
            winner=next(b for b in bs if b['name']==top)
            for b in bs:
                bpc=b.get('bass_pc'); nf = top if bpc is None or bpc==winner['root_pc'] else f"{top}/{NOTE[bpc]}"
                out.append({**b,'display':top,'root':winner['root'],
                            'suffix':winner['suffix'],'quality':winner['quality'],
                            'root_pc':winner['root_pc'],'name_full':nf})
        else:
            for b in bs: out.append({**b,'display':b['name']})
    return out

# ─── Tabulatură ───────────────────────────────────────────────────────────────
STD_NAMES  = ['E','A','D','G','B','e']
BASS_NAMES = ['E','A','D','G']
GVOI = {
    'C':[None,3,2,0,1,0],'Cm':[None,3,5,5,4,3],'D':[None,None,0,2,3,2],
    'Dm':[None,None,0,2,3,1],'E':[0,2,2,1,0,0],'Em':[0,2,2,0,0,0],
    'F':[1,1,2,3,3,1],'Fm':[1,1,3,3,2,1],'G':[3,2,0,0,0,3],'Gm':[3,5,5,3,3,3],
    'A':[None,0,2,2,2,0],'Am':[None,0,2,2,1,0],'B':[None,2,4,4,4,2],'Bm':[None,2,4,4,3,2],
    'F#':[2,2,3,4,4,2],'F#m':[2,2,4,4,3,2],'Bb':[None,1,3,3,3,1],'Bbm':[None,1,3,3,2,1],
    'G7':[3,2,0,0,0,1],'D7':[None,None,0,2,1,2],'A7':[None,0,2,0,2,0],'E7':[0,2,0,1,0,0],
    'Cmaj7':[None,3,2,0,0,0],'Gmaj7':[3,2,0,0,0,2],'Am7':[None,0,2,0,1,0],
    'Em7':[0,2,2,0,3,0],'Dm7':[None,None,0,2,1,1],'Bm7':[None,2,4,2,3,2],
}
BVOI = {
    'C':[None,3,2,0],'Cm':[None,3,1,0],'D':[None,None,0,2],'Dm':[None,None,0,2],
    'E':[0,2,2,1],'Em':[0,2,2,0],'F':[1,3,3,2],'G':[3,2,0,0],
    'A':[None,0,2,2],'Am':[None,0,2,2],'B':[None,2,4,4],'Bm':[None,2,4,4],
}
def _rpc(n):
    for ln in [2,1]:
        if n[:ln] in NOTE: return NOTE.index(n[:ln])
        if n[:ln] in NOTEF: return NOTEF.index(n[:ln])
    return 0
def get_tab(name, inst='Guitar'):
    voi=GVOI if inst=='Guitar' else BVOI
    if name in voi: return voi[name]
    root=name
    for s in ['maj7','m7','7','m','dim','aug','sus4','sus2']:
        if name.endswith(s): root=name[:-len(s)]; break
    if root in voi: return voi[root]
    pc=_rpc(root); f=(pc-5)%12; im='m' in name.lower().replace('maj','')
    if inst=='Guitar':
        return ([None,f,f+2,f+2,f+1,f] if im else [None,f,f+2,f+2,f+2,f]) if f>0 else ([None,0,2,2,1,0] if im else [None,0,2,2,2,0])
    return ([None,f,f+2,f+2] if f>0 else [None,0,2,2])

def bass_note_frets(bass_pc, max_fret=7):
    """Un singur fret aprins pe cele 4 corzi de bas (E A D G), pentru NOTA reala
    de bas detectata la acea bataie -- nu forma de acord, ci nota exacta cantata."""
    if bass_pc is None: return [None,None,None,None]
    open_pc = [4,9,2,7]  # E A D G
    best=None
    for si,opc in enumerate(open_pc):
        f=(bass_pc-opc)%12
        if best is None or f<best[1]: best=(si,f)
    frets=[None]*4; frets[best[0]]=best[1]
    return frets

# ─── Pian ──────────────────────────────────────────────────────────────────
PIANO_WHITE_PC = [0,2,4,5,7,9,11]   # C D E F G A B
def piano_pcs_for_chord(root_pc, suf):
    if root_pc is None: return []
    ivs,_,_ = CTMPL.get(suf, ([0,4,7],None,None))
    return sorted(set((root_pc+iv)%12 for iv in ivs))

# ─── Download YouTube ─────────────────────────────────────────────────────────
def download_yt(url, cb=None):
    import yt_dlp
    tmp=tempfile.mkdtemp(); title_h=[None]
    class Hook:
        def __call__(self,d):
            if d['status']=='downloading':
                tot=d.get('total_bytes') or d.get('total_bytes_estimate') or 1
                pct=d.get('downloaded_bytes',0)/tot
                if cb: cb(pct*0.65,f'Descărcare {int(pct*100)}%...')
            elif d['status']=='finished':
                if cb: cb(0.70,'Conversie audio...')
    opts={'format':'bestaudio/best','outtmpl':os.path.join(tmp,'%(title)s.%(ext)s'),
          'quiet':True,'no_warnings':True,
          'postprocessors':[{'key':'FFmpegExtractAudio','preferredcodec':'mp3','preferredquality':'192'}],
          'progress_hooks':[Hook()]}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info=ydl.extract_info(url,download=True); title_h[0]=info.get('title','Track')
    for f in os.listdir(tmp):
        if any(f.endswith(e) for e in ['.mp3','.m4a','.webm','.ogg','.wav']):
            return os.path.join(tmp,f), title_h[0]
    raise FileNotFoundError('Fișier audio negăsit după descărcare')

# ─── Aplicația principală ─────────────────────────────────────────────────────
class App:
    # Dimensiuni grid chordify
    RULER_H    = 22    # ruler cu timecode deasupra fiecărui rând
    ROW_H      = 88    # înălțimea blocurilor de acorduri
    ROW_PAD    = 12    # spațiu între rânduri
    LEFT_W     = 52    # margine stânga (nr. măsură)
    MIN_BW     = 48    # lățime minimă per bătaie (px)

    def __init__(self):
        self.root = ctk.CTk()
        self.root.title('ChordVision Pro')
        self.root.geometry('1320x900'); self.root.configure(fg_color=BG)
        self.root.minsize(900,650)
        self.analysis=None; self.track_name=''; self.audio_path=None
        self.playing=False; self.play_start=0.0; self.pause_pos=0.0
        self.duration=0.0; self._tick_job=None; self._cur_beat=-1; self._tab_cur_beat_pending=None
        self.inst_var=ctk.StringVar(value='Guitar')
        # canvas state
        self._cv=None; self._beat_rects={}; self._playhead_id=None
        self._cur_hl=-1; self._last_w=0
        self._beats_ref=[]; self._sig_ref=4; self._bpm_ref=120
        self._n_per_row=4; self._beat_w=60; self._meas_w=240; self._total_h=1000
        self._active_tab='Acorduri'
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self):
        top=ctk.CTkFrame(self.root,fg_color=SURF,corner_radius=0,height=50)
        top.pack(fill='x'); top.pack_propagate(False)
        ctk.CTkLabel(top,text='⬡  ChordVision Pro',font=ctk.CTkFont('Courier',17,'bold'),text_color=ACCENT2).pack(side='left',padx=18)
        ctk.CTkLabel(top,text='Acorduri per bătaie • Măsuri sincronizate',font=ctk.CTkFont(size=11),text_color=TEXT3).pack(side='left',padx=4)
        body=ctk.CTkFrame(self.root,fg_color=BG); body.pack(fill='both',expand=True)
        left=ctk.CTkFrame(body,fg_color=SURF,corner_radius=0,width=268)
        left.pack(side='left',fill='y'); left.pack_propagate(False); self._build_left(left)
        right=ctk.CTkFrame(body,fg_color=BG); right.pack(side='left',fill='both',expand=True)
        self._build_right(right)

    def _sep(self,p): ctk.CTkFrame(p,fg_color=BORDER,height=1).pack(fill='x',padx=12,pady=5)
    def _slbl(self,p,t): ctk.CTkLabel(p,text=t.upper(),font=ctk.CTkFont(size=9,weight='bold'),text_color=TEXT3).pack(anchor='w',padx=12,pady=(12,2))

    def _build_left(self,p):
        ctk.CTkLabel(p,text='Sursă audio',font=ctk.CTkFont(size=13,weight='bold'),text_color=TEXT).pack(anchor='w',padx=12,pady=(16,2))
        self._sep(p)
        self._slbl(p,'YouTube')
        self.yt_entry=ctk.CTkEntry(p,placeholder_text='https://youtube.com/watch?v=...',fg_color=SURF2,border_color=BORDER,text_color=TEXT,width=244)
        self.yt_entry.pack(padx=12,pady=(0,6))
        ctk.CTkButton(p,text='⬇  Descarcă & Analizează',fg_color=ACCENT,hover_color=ACCENT2,text_color='white',font=ctk.CTkFont(size=12,weight='bold'),command=self._yt,height=36).pack(padx=12,fill='x',pady=(0,4))
        self._sep(p)
        self._slbl(p,'Fișier local')
        ctk.CTkButton(p,text='📂  Alege fișier audio',fg_color=SURF2,hover_color=SURF3,text_color=TEXT,border_color=BORDER,border_width=1,command=self._file,height=36).pack(padx=12,fill='x')
        self.file_lbl=ctk.CTkLabel(p,text='Niciun fișier',font=ctk.CTkFont(size=10),text_color=TEXT3,wraplength=240)
        self.file_lbl.pack(padx=12,anchor='w',pady=(3,0))
        self._sep(p)
        self._slbl(p,'Progres')
        self.pbar=ctk.CTkProgressBar(p,fg_color=SURF2,progress_color=ACCENT,width=244)
        self.pbar.pack(padx=12,pady=(0,4)); self.pbar.set(0)
        self.stat_lbl=ctk.CTkLabel(p,text='Pregătit.',font=ctk.CTkFont(size=11),text_color=TEXT2)
        self.stat_lbl.pack(padx=12,anchor='w')
        self._sep(p)
        self._slbl(p,'Tabulatură')
        for inst in ['Guitar','Bass','Pian']:
            ctk.CTkRadioButton(p,text=inst,variable=self.inst_var,value=inst,fg_color=ACCENT,text_color=TEXT,command=self._on_inst).pack(anchor='w',padx=20,pady=2)
        self._sep(p)
        self._slbl(p,'Rezultate')
        cf=ctk.CTkFrame(p,fg_color='transparent'); cf.pack(padx=8,fill='x')
        self._cards={}
        for i,(k,lbl) in enumerate([('key','Tonalitate'),('mode','Mod'),('bpm','BPM'),('sig','Metru'),('dur','Durată')]):
            c=ctk.CTkFrame(cf,fg_color=SURF2,corner_radius=7)
            c.grid(row=i//2,column=i%2,padx=3,pady=3,sticky='ew')
            cf.grid_columnconfigure(0,weight=1); cf.grid_columnconfigure(1,weight=1)
            ctk.CTkLabel(c,text=lbl,font=ctk.CTkFont(size=8),text_color=TEXT3).pack(anchor='w',padx=7,pady=(5,0))
            v=ctk.CTkLabel(c,text='—',font=ctk.CTkFont('Courier',14,'bold'),text_color=ACCENT2)
            v.pack(anchor='w',padx=7,pady=(0,5)); self._cards[k]=v
        self._sep(p)
        self._slbl(p,'Legendă')
        for lbl,col in [('I — Tonică',ACCENT),('IV — Subdominantă',GREEN),('V — Dominantă',YELLOW),('ii / vi',CYAN),('iii','#60a5fa'),('VII / dim',RED)]:
            r=ctk.CTkFrame(p,fg_color='transparent'); r.pack(fill='x',padx=12,pady=1)
            ctk.CTkLabel(r,text='●',font=ctk.CTkFont(size=11),text_color=col).pack(side='left')
            ctk.CTkLabel(r,text=lbl,font=ctk.CTkFont(size=10),text_color=TEXT2).pack(side='left',padx=5)

    def _build_right(self,p):
        pb=ctk.CTkFrame(p,fg_color=SURF2,corner_radius=12)
        pb.pack(fill='x',padx=12,pady=(10,4))
        r1=ctk.CTkFrame(pb,fg_color='transparent'); r1.pack(fill='x',padx=12,pady=(10,4))
        self.play_btn=ctk.CTkButton(r1,text='▶',width=44,height=44,corner_radius=22,fg_color=ACCENT,hover_color=ACCENT2,text_color='white',font=ctk.CTkFont(size=20),command=self._toggle_play)
        self.play_btn.pack(side='left',padx=(0,8))
        ctk.CTkButton(r1,text='⏮',width=34,height=34,corner_radius=17,fg_color=SURF3,hover_color=BORDER,text_color=TEXT2,font=ctk.CTkFont(size=13),command=self._restart).pack(side='left',padx=2)
        self.track_lbl=ctk.CTkLabel(r1,text='Nicio melodie',font=ctk.CTkFont(size=12,weight='bold'),text_color=TEXT)
        self.track_lbl.pack(side='left',padx=12)
        self.time_lbl=ctk.CTkLabel(r1,text='0:00 / 0:00',font=ctk.CTkFont('Courier',11),text_color=TEXT2)
        self.time_lbl.pack(side='right')
        self.info_lbl=ctk.CTkLabel(r1,text='',font=ctk.CTkFont(size=10),text_color=TEXT3)
        self.info_lbl.pack(side='right',padx=10)
        self.seek=ctk.CTkSlider(pb,from_=0,to=100,command=self._on_seek,fg_color=SURF3,progress_color=ACCENT,button_color=ACCENT2)
        self.seek.pack(fill='x',padx=12,pady=(0,10)); self.seek.set(0)
        tabrow=ctk.CTkFrame(p,fg_color='transparent'); tabrow.pack(fill='x',padx=12,pady=(4,2))
        self._tab_btns={}
        for t in ['Acorduri','Tabulatură','Cerc cvinte']:
            b=ctk.CTkButton(tabrow,text=t,width=112,height=30,fg_color=ACCENT if t=='Acorduri' else SURF2,hover_color=SURF3,text_color='white' if t=='Acorduri' else TEXT2,font=ctk.CTkFont(size=11),corner_radius=8,command=lambda x=t:self._switch_tab(x))
            b.pack(side='left',padx=2); self._tab_btns[t]=b
        self.content=ctk.CTkFrame(p,fg_color=BG)
        self.content.pack(fill='both',expand=True,padx=12,pady=(2,8))
        self._show_ph()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    def _switch_tab(self,name):
        self._active_tab=name
        for t,b in self._tab_btns.items():
            b.configure(fg_color=ACCENT if t==name else SURF2,text_color='white' if t==name else TEXT2)
        self._refresh()

    def _refresh(self):
        for w in self.content.winfo_children(): w.destroy()
        self._cv=None; self._beat_rects={}; self._playhead_id=None; self._cur_hl=-1; self._last_w=0
        if not self.analysis: self._show_ph(); return
        if   self._active_tab=='Acorduri':    self._build_chordify()
        elif self._active_tab=='Tabulatură':  self._build_tab()
        elif self._active_tab=='Cerc cvinte': self._build_cof()
        if self._cur_beat>=0:
            if self._active_tab=='Acorduri' and self._cv: self._cv_hl(self._cur_beat)
            elif self._active_tab=='Tabulatură': self._hl_tab(self._cur_beat)

    def _show_ph(self):
        ctk.CTkLabel(self.content,text='Încarcă un fișier sau URL YouTube ♪',font=ctk.CTkFont(size=14),text_color=TEXT3).pack(expand=True)

    # ── CHORDIFY VIEW ─────────────────────────────────────────────────────────
    def _build_chordify(self):
        outer=tk.Frame(self.content,bg=BG); outer.pack(fill='both',expand=True)
        # legend
        leg=tk.Frame(outer,bg=SURF2,height=24); leg.pack(fill='x'); leg.pack_propagate(False)
        for lbl,col in [('I',ACCENT),('IV',GREEN),('V',YELLOW),('ii/vi',CYAN),('iii','#60a5fa'),('VII/dim',RED)]:
            tk.Label(leg,text=f'● {lbl}',fg=col,bg=SURF2,font=('Courier',9)).pack(side='left',padx=8)
        tk.Label(leg,text='Click = sari la acel moment | Scroll = derulare',fg=TEXT3,bg=SURF2,font=('Arial',8,'italic')).pack(side='right',padx=10)
        # canvas
        frame=tk.Frame(outer,bg=BG); frame.pack(fill='both',expand=True)
        cv=tk.Canvas(frame,bg=BG,highlightthickness=0,cursor='hand2')
        sb=ctk.CTkScrollbar(frame,command=cv.yview)
        sb.pack(side='right',fill='y'); cv.pack(side='left',fill='both',expand=True)
        cv.configure(yscrollcommand=sb.set)
        self._cv=cv
        self._beats_ref=self.analysis['beats']; self._sig_ref=self.analysis['time_sig']; self._bpm_ref=self.analysis['bpm']
        cv.bind('<MouseWheel>',lambda e:cv.yview_scroll(int(-1*(e.delta/120)),'units'))
        cv.bind('<Button-4>',lambda e:cv.yview_scroll(-1,'units'))
        cv.bind('<Button-5>',lambda e:cv.yview_scroll(1,'units'))
        cv.bind('<Button-1>',self._cv_click)
        cv.bind('<Configure>',lambda e:self._cv_draw(e.width))

    def _cv_draw(self,w):
        if w==self._last_w or w<50: return
        self._last_w=w
        cv=self._cv; beats=self._beats_ref; sig=self._sig_ref
        cv.delete('all'); self._beat_rects={}; self._playhead_id=None; self._cur_hl=-1

        # Calculate layout
        usable     = w - self.LEFT_W - 8
        meas_w_min = sig * self.MIN_BW
        n_per_row  = max(1, usable // meas_w_min)
        # Distribute space evenly
        meas_w     = usable // n_per_row
        beat_w     = meas_w // sig
        meas_w     = beat_w * sig   # re-align

        self._n_per_row=n_per_row; self._beat_w=beat_w; self._meas_w=meas_w

        n_meas     = self.analysis['n_measures']
        n_rows     = math.ceil(n_meas/n_per_row)
        row_stride = self.RULER_H + self.ROW_H + self.ROW_PAD
        total_h    = self.ROW_PAD + n_rows*row_stride + 20
        self._total_h=total_h

        X0=self.LEFT_W
        by_m={}
        for b in beats: by_m.setdefault(b['measure'],[]).append(b)

        FNT_ROOT = ('Courier',14,'bold')
        FNT_SUF  = ('Courier', 9)
        FNT_BEAT = ('Courier', 8)
        FNT_TIME = ('Courier', 8)
        FNT_MEAS = ('Courier', 9,'bold')

        for row in range(n_rows):
            ry = self.ROW_PAD + row*row_stride

            # Row bg
            cv.create_rectangle(0,ry,w,ry+self.RULER_H+self.ROW_H,fill=SURF,outline='')

            # Left margin – row time
            first_meas = row*n_per_row
            if first_meas < n_meas:
                fmb = by_m.get(first_meas,[])
                t0  = fmb[0]['time'] if fmb else 0.0
                cv.create_text(2,ry+self.RULER_H//2,text=self._ft(t0),anchor='w',fill=TEXT3,font=FNT_TIME)

            for mi_row in range(n_per_row):
                meas_idx = row*n_per_row + mi_row
                if meas_idx >= n_meas: break
                mx = X0 + mi_row*meas_w
                mbs = by_m.get(meas_idx,[])

                # Measure number
                cv.create_text(mx+2,ry+self.RULER_H//2,text=f'm{meas_idx+1}',anchor='w',fill=TEXT3,font=FNT_MEAS)

                # Time of last beat in measure
                if mbs:
                    cv.create_text(mx+meas_w-2,ry+self.RULER_H//2,text=self._ft(mbs[-1]['time']),anchor='e',fill=TEXT3,font=FNT_TIME)

                # Barline
                cv.create_line(mx,ry,mx,ry+self.RULER_H+self.ROW_H,fill=BORDER2,width=1)

                cy=ry+self.RULER_H

                for bi,b in enumerate(mbs[:sig]):
                    bx=mx+bi*beat_w
                    col=b.get('color',ACCENT)
                    fil=b.get('fill',SURF2)

                    # Beat separator (thin vertical line)
                    if bi>0:
                        cv.create_line(bx,cy+6,bx,cy+self.ROW_H-6,fill=BORDER,width=1,dash=(2,3))

                    # Block
                    rid=cv.create_rectangle(bx+2,cy+4,bx+beat_w-2,cy+self.ROW_H-4,
                                             fill=fil,outline=col,width=1,
                                             tags=('beat',f'b{b["beat_idx"]}'))

                    # Beat number (1-4) top inside block
                    cv.create_text(bx+beat_w//2,cy+13,text=str(bi+1),
                                   fill=TEXT3,font=FNT_BEAT,anchor='center')

                    # Root (big)
                    if beat_w>=32:
                        rid2=cv.create_text(bx+beat_w//2,cy+self.ROW_H*0.43,
                                            text=b['root'],fill=col,font=FNT_ROOT,
                                            anchor='center',tags=(f'b{b["beat_idx"]}',))
                    else:
                        rid2=None

                    # Suffix (small)
                    suf=b['suffix'] if b['suffix'] else 'maj'
                    if beat_w>=32:
                        rid3=cv.create_text(bx+beat_w//2,cy+self.ROW_H*0.73,
                                            text=suf,fill=TEXT3,font=FNT_SUF,
                                            anchor='center',tags=(f'b{b["beat_idx"]}',))
                    else:
                        rid3=None

                    self._beat_rects[b['beat_idx']]=(rid,rid2,rid3,col,fil)

        # End barline
        cv.create_line(X0,self.ROW_PAD,X0,self.ROW_PAD+(n_rows-1)*row_stride+self.RULER_H+self.ROW_H,fill=BORDER2,width=1)

        # Playhead
        self._playhead_id=cv.create_line(X0,0,X0,total_h,fill=WHITE,width=2,dash=(5,3),tags='playhead')
        cv.tag_raise('playhead')

        cv.configure(scrollregion=(0,0,w,total_h))

        # Restore
        if self._cur_beat>=0: self._cv_hl(self._cur_beat); self._cv_ph(self.pause_pos)

    def _cv_click(self,event):
        cv=self._cv
        if not cv or not self.analysis: return
        items=cv.find_closest(event.x,cv.canvasy(event.y))
        for item in items:
            for tag in cv.gettags(item):
                if tag.startswith('b') and tag[1:].isdigit():
                    idx=int(tag[1:])
                    if idx<len(self._beats_ref):
                        self._seek_to(self._beats_ref[idx]['time']); return
        # fallback: position seek
        cx=event.x; cy_c=cv.canvasy(event.y)
        if self._n_per_row<1: return
        row_stride=self.RULER_H+self.ROW_H+self.ROW_PAD
        row=int(cy_c//row_stride)
        rel_x=cx-self.LEFT_W
        if rel_x<0: return
        mi_row=int(rel_x//self._meas_w)
        bi_f  =int((rel_x-mi_row*self._meas_w)//self._beat_w)
        meas  =row*self._n_per_row+mi_row
        beat_t=meas*self._sig_ref+max(0,min(bi_f,self._sig_ref-1))
        if beat_t<len(self._beats_ref): self._seek_to(self._beats_ref[beat_t]['time'])

    def _cv_hl(self,beat_idx):
        cv=self._cv
        if not cv: return
        # restore prev
        if self._cur_hl>=0 and self._cur_hl in self._beat_rects:
            rid,_,_,col,fil=self._beat_rects[self._cur_hl]
            cv.itemconfig(rid,fill=fil,outline=col,width=1)
        self._cur_hl=beat_idx
        if beat_idx in self._beat_rects:
            rid,r2,r3,col,fil=self._beat_rects[beat_idx]
            bright=_tint(col,0.55)
            cv.itemconfig(rid,fill=bright,outline=WHITE,width=2)
            cv.tag_raise('playhead')
            if r2: cv.tag_raise(r2)
            if r3: cv.tag_raise(r3)
            self._cv_scroll(beat_idx)

    def _cv_scroll(self,beat_idx):
        if beat_idx>=len(self._beats_ref) or self._n_per_row<1: return
        meas=self._beats_ref[beat_idx]['measure']
        row=meas//self._n_per_row
        row_stride=self.RULER_H+self.ROW_H+self.ROW_PAD
        ry=self.ROW_PAD+row*row_stride
        ry2=ry+self.RULER_H+self.ROW_H
        tot=self._total_h
        top,bot=ry/tot,ry2/tot
        ct,cb=self._cv.yview()
        if top<ct: self._cv.yview_moveto(max(0,top-0.02))
        elif bot>cb: self._cv.yview_moveto(min(1,top-0.04))

    def _cv_ph(self,pos):
        cv=self._cv
        if not cv or not self._playhead_id or self._n_per_row<1: return
        beat_dur=60.0/self._bpm_ref
        beat_f=pos/beat_dur
        meas=int(beat_f//self._sig_ref)
        bi=beat_f%self._sig_ref
        n_meas=self.analysis['n_measures']
        if meas>=n_meas: return
        row=meas//self._n_per_row
        mi=meas%self._n_per_row
        row_stride=self.RULER_H+self.ROW_H+self.ROW_PAD
        ry=self.ROW_PAD+row*row_stride
        px=self.LEFT_W+mi*self._meas_w+bi*self._beat_w
        cv.coords(self._playhead_id,px,ry,px,ry+self.RULER_H+self.ROW_H)
        cv.tag_raise('playhead')

    # ── Tabulatură ────────────────────────────────────────────────────────────
    def _build_tab(self):
        a=self.analysis; inst=self.inst_var.get()
        is_piano = inst=='Pian'
        snames = [] if is_piano else (STD_NAMES if inst=='Guitar' else BASS_NAMES)
        n_str  = len(snames)
        outer=ctk.CTkFrame(self.content,fg_color=BG); outer.pack(fill='both',expand=True)
        sc=tk.Canvas(outer,bg=BG,highlightthickness=0)
        sb=ctk.CTkScrollbar(outer,command=sc.yview); sb.pack(side='right',fill='y')
        sc.pack(side='left',fill='both',expand=True); sc.configure(yscrollcommand=sb.set)
        inner=ctk.CTkFrame(sc,fg_color=BG)
        win=sc.create_window((0,0),window=inner,anchor='nw')
        inner.bind('<Configure>',lambda e:sc.configure(scrollregion=sc.bbox('all')))
        sc.bind('<Configure>',lambda e:sc.itemconfig(win,width=e.width))
        sc.bind_all('<MouseWheel>',lambda e:sc.yview_scroll(int(-1*(e.delta/120)),'units'))
        self._tab_sc=sc; self._tab_inner=inner
        self._tab_cells={}     # beat_idx -> (cell_frame, base_color)
        self._tab_rows=[]      # (row_frame, first_measure_in_row)
        self._tab_cur_beat=-1
        hdr=ctk.CTkFrame(inner,fg_color=SURF,corner_radius=8); hdr.pack(fill='x',padx=8,pady=(8,4))
        ctk.CTkLabel(hdr,text=f'Tabulatură {inst}  •  {NOTE[a["key_root"]]} {a["key_mode"].capitalize()}  •  {a["bpm"]} BPM  •  {a["time_sig"]}/4',font=ctk.CTkFont(size=12,weight='bold'),text_color=TEXT).pack(side='left',padx=12,pady=8)
        if not is_piano:
            ctk.CTkLabel(hdr,text='Acordaj: '+' '.join(snames[::-1]),font=ctk.CTkFont('Courier',10),text_color=TEXT2).pack(side='right',padx=12)
        by_m={}
        for b in a['beats']: by_m.setdefault(b['measure'],[]).append(b)
        cell_w = 128 if is_piano else 66
        MEAS_PER_ROW = max(1, min(4 if is_piano else 6, a.get('time_sig',4) and 4))
        n_measures=a['n_measures']
        for ri in range(math.ceil(n_measures/MEAS_PER_ROW)):
            rf=ctk.CTkFrame(inner,fg_color=SURF,corner_radius=8); rf.pack(fill='x',padx=8,pady=3)
            self._tab_rows.append((rf, ri*MEAS_PER_ROW))
            for mi in range(ri*MEAS_PER_ROW,min((ri+1)*MEAS_PER_ROW,n_measures)):
                mbs=by_m.get(mi,[])
                if not mbs: continue
                mf=ctk.CTkFrame(rf,fg_color=BG,corner_radius=6); mf.pack(side='left',padx=(6,2),pady=6)
                ctk.CTkLabel(mf,text=f'm{mi+1}',font=ctk.CTkFont(size=8),text_color=TEXT3).pack(anchor='w',padx=2)
                bf=ctk.CTkFrame(mf,fg_color=BG); bf.pack()
                for b in mbs:
                    col=b.get('color',ACCENT)
                    label = b.get('name_full', b['name'])
                    cf=ctk.CTkFrame(bf,fg_color=SURF2,corner_radius=7,border_color=col,border_width=1)
                    cf.pack(side='left',padx=2,pady=0)
                    ctk.CTkLabel(cf,text=label,font=ctk.CTkFont('Courier',11,'bold'),text_color=col,width=cell_w-8).pack(pady=(3,0))
                    if is_piano:
                        pcs=piano_pcs_for_chord(b.get('root_pc'), b.get('suffix',''))
                        diag=tk.Canvas(cf,bg=SURF2,highlightthickness=0,width=118,height=70)
                        diag.pack(padx=4,pady=(0,4)); self._draw_piano(diag,pcs,b.get('root_pc'),col)
                    elif inst=='Bass':
                        frets=bass_note_frets(b.get('bass_pc'))
                        diag=tk.Canvas(cf,bg=SURF2,highlightthickness=0,width=cell_w-14,height=n_str*20+8)
                        diag.pack(padx=4,pady=(0,4)); self._draw_fb(diag,frets,snames,col)
                    else:
                        frets=get_tab(b['name'],inst)
                        diag=tk.Canvas(cf,bg=SURF2,highlightthickness=0,width=cell_w-14,height=n_str*20+8)
                        diag.pack(padx=4,pady=(0,4)); self._draw_fb(diag,frets,snames,col)
                    self._tab_cells[b['beat_idx']]=(cf,col)
                    for ww in [cf,diag]: ww.bind('<Button-1>',lambda e,t=b['time']:self._seek_to(t))
        inner.update_idletasks()
        self._tab_row_y=[rf.winfo_y() for rf,_ in self._tab_rows]
        self._tab_total_h=max(inner.winfo_height(),1)
        self._tab_meas_per_row=MEAS_PER_ROW
        if self._tab_cur_beat_pending is not None:
            self._hl_tab(self._tab_cur_beat_pending)

    def _draw_fb(self,cv,frets,snames,col):
        W=60; real=[f for f in frets if f is not None and f>0]
        mn=min(real) if real else 0; nr=max((max(real)-mn+1) if real else 0,3)
        fw=(W-18)/max(nr,1); H=len(snames)*20+8; cv.configure(height=H)
        if mn>0: cv.create_text(W-2,8,text=str(mn),fill=TEXT3,font=('Courier',7),anchor='e')
        for i,(sn,f) in enumerate(zip(snames,frets)):
            y=10+i*20; cv.create_line(12,y,W-4,y,fill=BORDER2,width=1)
            cv.create_text(8,y,text=sn,fill=TEXT3,font=('Courier',7),anchor='e')
            if f is None: cv.create_text(W-8,y,text='×',fill=RED,font=('Courier',9,'bold'))
            elif f==0: cv.create_oval(W-14,y-4,W-6,y+4,outline=GREEN,width=1)
            else:
                rel=f-mn; x=14+rel*fw+fw/2
                cv.create_oval(x-5,y-5,x+5,y+5,fill=col,outline='')

    def _draw_piano(self,cv,pcs,root_pc,col):
        W=118; H=70; n_white=7
        ww=W/n_white
        cv.configure(width=W,height=H)
        white_x={}
        for i,pc in enumerate(PIANO_WHITE_PC):
            x0=i*ww; on = pc in pcs
            fill = col if on else SURF3
            cv.create_rectangle(x0+1,1,x0+ww-1,H-1,fill=fill,outline=BORDER2,width=1)
            white_x[pc]=x0
            if pc==root_pc:
                cv.create_oval(x0+ww/2-4,H-14,x0+ww/2+4,H-6,fill=WHITE,outline='')
        black=[1,3,None,6,8,10,None]  # pc pentru negre dupa fiecare alba, None=fara neagra
        bw=ww*0.6; bh=H*0.62
        for i,pc in enumerate(black):
            if pc is None: continue
            x0=white_x[PIANO_WHITE_PC[i]]+ww-bw/2
            on = pc in pcs
            fill = col if on else SURF
            cv.create_rectangle(x0,1,x0+bw,bh,fill=fill,outline=BORDER,width=1)
            if pc==root_pc:
                cv.create_oval(x0+bw/2-3,bh-11,x0+bw/2+3,bh-5,fill=WHITE,outline='')

    # ── Cerc cvinte ───────────────────────────────────────────────────────────
    def _build_cof(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        a=self.analysis; beats=a['beats']
        usage=Counter(b['root_pc'] for b in beats); maxu=max(usage.values(),default=1)
        chord_freq=Counter(b['name'] for b in beats)
        fig,axes=plt.subplots(1,2,figsize=(11,5.5),facecolor=BG)
        fig.subplots_adjust(left=0.02,right=0.98,top=0.92,bottom=0.06,wspace=0.15)
        ax=axes[0]; ax.set_facecolor(BG); ax.axis('off'); ax.set_aspect('equal')
        ax.set_title('Cercul cvintelor',color=TEXT,fontsize=12,pad=6)
        R,ri=1.0,0.56
        for i,pc in enumerate(COF):
            a1=(i/12)*2*np.pi-np.pi/2-np.pi/12; a2=a1+2*np.pi/12
            angs=np.linspace(a1,a2,20)
            xo,yo=np.cos(angs)*R,np.sin(angs)*R; xi,yi=np.cos(angs[::-1])*ri,np.sin(angs[::-1])*ri
            uf=usage.get(pc,0)/maxu
            col=(ACCENT if a['key_mode']=='major' else GREEN) if pc==a['key_root'] else (ACCENT if uf>0 else SURF2)
            alpha=1.0 if pc==a['key_root'] else (0.25+uf*0.55 if uf>0 else 1.0)
            ax.fill(np.concatenate([xo,xi]),np.concatenate([yo,yi]),color=col,alpha=alpha,zorder=1)
            ax.plot(np.append(np.concatenate([xo,xi]),xo[0]),np.append(np.concatenate([yo,yi]),yo[0]),color=BG,linewidth=1.5,zorder=2)
            ma=(a1+a2)/2; ax.text(np.cos(ma)*(R+ri)/2,np.sin(ma)*(R+ri)/2,NOTE[pc],color='white' if (pc==a['key_root'] or uf>0.2) else TEXT3,fontsize=10 if pc==a['key_root'] else 8,fontweight='bold' if pc==a['key_root'] else 'normal',ha='center',va='center',zorder=3)
        ax.add_patch(plt.Circle((0,0),ri*0.95,color=SURF,zorder=4))
        ax.text(0,0.1,NOTE[a['key_root']]+('m' if a['key_mode']=='minor' else ''),color=ACCENT2,fontsize=18,fontweight='bold',ha='center',va='center',zorder=5)
        ax.text(0,-0.15,a['key_mode'].capitalize(),color=TEXT3,fontsize=10,ha='center',va='center',zorder=5)
        ax.set_xlim(-1.15,1.15); ax.set_ylim(-1.15,1.15)
        ax2=axes[1]; ax2.set_facecolor(SURF); ax2.set_title('Frecvența acordurilor',color=TEXT,fontsize=12,pad=6)
        top=chord_freq.most_common(12)
        if top:
            names,cnts=zip(*top); total=sum(cnts); pcts=[c/total*100 for c in cnts]
            qmap={b['name']:b.get('quality','major') for b in beats}
            colors=[QCOL.get(qmap.get(n,'major'),ACCENT) for n in names]
            bars=ax2.barh(range(len(names)),pcts,color=colors,alpha=0.85,height=0.6)
            ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names,color=TEXT,fontsize=10,fontfamily='monospace')
            ax2.set_xlabel('Frecvență (%)',color=TEXT2,fontsize=9); ax2.tick_params(colors=TEXT3,labelsize=8)
            ax2.set_xlim(0,max(pcts)*1.25)
            for bar,pct in zip(bars,pcts):
                ax2.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,f'{pct:.1f}%',color=TEXT2,va='center',fontsize=8)
        for sp in ax2.spines.values(): sp.set_color(BORDER)
        fc=FigureCanvasTkAgg(fig,master=self.content); fc.draw(); fc.get_tk_widget().pack(fill='both',expand=True); plt.close(fig)

    # ── Playback ──────────────────────────────────────────────────────────────
    def _toggle_play(self):
        if not self.audio_path: return
        if self.playing: self._pause()
        else: self._play()

    def _play(self):
        if not HAS_PYGAME or not self.audio_path: return
        try:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(self.audio_path)
                pygame.mixer.music.play(start=self.pause_pos)
            else:
                pygame.mixer.music.unpause()
            self.play_start=time.time()-self.pause_pos; self.playing=True
            self.play_btn.configure(text='⏸'); self._tick()
        except Exception as e: messagebox.showerror('Playback eroare',str(e))

    def _pause(self):
        if not HAS_PYGAME: return
        pygame.mixer.music.pause()
        self.pause_pos=time.time()-self.play_start; self.playing=False
        self.play_btn.configure(text='▶')
        if self._tick_job: self.root.after_cancel(self._tick_job); self._tick_job=None

    def _restart(self):
        self._pause(); self.pause_pos=0.0; self.seek.set(0)
        self.time_lbl.configure(text=f'0:00 / {self._ft(self.duration)}')
        self._update_ui(0.0)

    def _seek_to(self,pos):
        was=self.playing; self._pause()
        self.pause_pos=max(0,min(pos,self.duration-0.05))
        self._update_ui(self.pause_pos)
        if was: self._play()

    def _on_seek(self,val):
        if not self.duration: return
        pos=float(val)/100*self.duration; was=self.playing
        self._pause(); self.pause_pos=pos; self._update_ui(pos)
        if was: self._play()

    def _tick(self):
        if not self.playing: return
        pos=time.time()-self.play_start
        if pos>=self.duration: self._pause(); self.pause_pos=0.0; return
        self._update_ui(pos)
        self._tick_job=self.root.after(50,self._tick)

    def _update_ui(self,pos):
        if not self.analysis: return
        self.time_lbl.configure(text=f'{self._ft(pos)} / {self._ft(self.duration)}')
        if self.duration>0: self.seek.set(pos/self.duration*100)
        beats=self.analysis['beats']
        idx=0
        for i,b in enumerate(beats):
            if b['time']<=pos: idx=i
            else: break
        if self._active_tab=='Acorduri' and self._cv:
            self._cv_ph(pos)
        if idx!=self._cur_beat:
            self._cur_beat=idx
            if self._active_tab=='Acorduri' and self._cv: self._cv_hl(idx)
            elif self._active_tab=='Tabulatură': self._hl_tab(idx)

    def _hl_tab(self,beat_idx):
        if not hasattr(self,'_tab_cells') or not self._tab_cells:
            self._tab_cur_beat_pending=beat_idx; return
        self._tab_cur_beat_pending=None
        if self._tab_cur_beat in self._tab_cells:
            pcf,pcol=self._tab_cells[self._tab_cur_beat]
            pcf.configure(fg_color=SURF2,border_color=pcol,border_width=1)
        self._tab_cur_beat=beat_idx
        if beat_idx in self._tab_cells:
            cf,col=self._tab_cells[beat_idx]
            cf.configure(fg_color='#2a2060',border_color=WHITE,border_width=2)
            self._tab_scroll_to(beat_idx)

    def _tab_scroll_to(self,beat_idx):
        if not getattr(self,'_tab_row_y',None): return
        meas=self.analysis['beats'][beat_idx]['measure']
        row=meas//max(self._tab_meas_per_row,1)
        if row>=len(self._tab_row_y): return
        tot=self._tab_total_h
        ry=self._tab_row_y[row]
        top,bot=ry/tot,(ry+140)/tot
        ct,cb=self._tab_sc.yview()
        if top<ct: self._tab_sc.yview_moveto(max(0,top-0.02))
        elif bot>cb: self._tab_sc.yview_moveto(min(1,top-0.05))

    # ── Analysis pipeline ─────────────────────────────────────────────────────
    def _set_prog(self,v,msg):
        self.root.after(0,lambda:self.pbar.set(v))
        self.root.after(0,lambda:self.stat_lbl.configure(text=msg))

    def _finish(self,a,tname,apath):
        self.analysis=a; self.track_name=tname; self.audio_path=apath
        self.duration=a['duration']; self.pause_pos=0.0; self._cur_beat=-1
        def _do():
            self._cards['key'].configure(text=NOTE[a['key_root']])
            self._cards['mode'].configure(text=a['key_mode'].capitalize())
            self._cards['bpm'].configure(text=str(a['bpm']))
            self._cards['sig'].configure(text=f'{a["time_sig"]}/4')
            self._cards['dur'].configure(text=self._ft(a['duration']))
            self.track_lbl.configure(text=tname[:52])
            self.time_lbl.configure(text=f'0:00 / {self._ft(self.duration)}')
            self.info_lbl.configure(text=f'{NOTE[a["key_root"]]} {a["key_mode"].capitalize()}  •  {a["bpm"]} BPM  •  {a["time_sig"]}/4')
            self._refresh()
        self.root.after(0,_do)

    def _run_file(self,path):
        import librosa
        try:
            self._set_prog(0.02,'Citire fișier...')
            y,sr=librosa.load(path,sr=None,mono=True)
            a=analyze(y,sr,self._set_prog); self._finish(a,Path(path).stem,path)
        except Exception as e:
            self._set_prog(0,f'Eroare: {e}')
            self.root.after(0,lambda:messagebox.showerror('Eroare',str(e)))

    def _run_yt(self,url):
        import librosa
        try:
            self._set_prog(0.01,'Conectare YouTube...')
            path,title=download_yt(url,self._set_prog)
            self._set_prog(0.75,'Decodare audio...')
            y,sr=librosa.load(path,sr=None,mono=True)
            a=analyze(y,sr,self._set_prog); self._finish(a,title,path)
        except Exception as e:
            self._set_prog(0,f'Eroare: {e}')
            self.root.after(0,lambda:messagebox.showerror('Eroare',str(e)))

    def _file(self):
        path=filedialog.askopenfilename(title='Selectează audio',
            filetypes=[('Audio','*.mp3 *.wav *.ogg *.flac *.m4a *.aac'),('Toate','*.*')])
        if not path: return
        self.file_lbl.configure(text=Path(path).name[:36])
        threading.Thread(target=self._run_file,args=(path,),daemon=True).start()

    def _yt(self):
        url=self.yt_entry.get().strip()
        if not url: messagebox.showwarning('URL lipsă','Introdu un URL YouTube.'); return
        threading.Thread(target=self._run_yt,args=(url,),daemon=True).start()

    def _on_inst(self):
        if self._active_tab=='Tabulatură': self._refresh()

    def _ft(self,s): return f'{int(s//60)}:{int(s%60):02d}'

    def run(self):
        self.root.protocol('WM_DELETE_WINDOW',self._quit); self.root.mainloop()

    def _quit(self):
        if HAS_PYGAME:
            try: pygame.mixer.music.stop(); pygame.quit()
            except: pass
        self.root.destroy()

if __name__=='__main__':
    App().run()
