# -*- coding: utf-8 -*-
"""Generate high-quality, publication-grade TikZ figures for all papers."""
from pathlib import Path

FIG_DIR = Path("paper_v2/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Figure 1: DaT-SPECT Striatal Uptake Schematic
# -------------------------------------------------------------
FIG_SCHEMATIC = r"""\begin{tikzpicture}[scale=0.92, every node/.style={transform shape}]
  \tikzset{
    skull/.style={fill=gray!7, draw=gray!45, line width=1.1pt},
    striatum/.style={fill=orange!85!red, draw=red!70!black, line width=0.6pt},
    core/.style={fill=yellow!95!orange, draw=none}
  }

  % (a) Normal Scan
  \begin{scope}[local bounding box=normalscan]
    \draw[skull] (0,0) ellipse (2.05cm and 2.25cm);
    \draw[dashed, gray!30, thin] (0,-2.0) -- (0,2.0);
    
    % Left Striatum
    \filldraw[striatum] (-0.65,0.45) circle (0.34cm);
    \filldraw[striatum, rotate around={25:(-0.95,-0.25)}] (-0.95,-0.25) ellipse (0.50cm and 0.21cm);
    
    % Right Striatum
    \filldraw[striatum] (0.65,0.45) circle (0.34cm);
    \filldraw[striatum, rotate around={-25:(0.95,-0.25)}] (0.95,-0.25) ellipse (0.50cm and 0.21cm);
    
    % Caudate hot cores
    \filldraw[core] (-0.65,0.45) circle (0.17cm);
    \filldraw[core] (0.65,0.45) circle (0.17cm);
    
    % Labels (placed well away from center)
    \draw[-{Latex[length=1.8mm]}, thick, gray!60!black] (-1.8,1.2) node[above, font=\scriptsize\bfseries] {Caudate Head} -- (-0.85,0.6);
    \draw[-{Latex[length=1.8mm]}, thick, gray!60!black] (-1.8,-1.2) node[below, font=\scriptsize\bfseries] {Putamen Tail} -- (-1.1,-0.45);
    
    \node[font=\small\bfseries, align=center] at (0,-2.7) {(a) Normal Uptake\\{\footnotesize Symmetric bilateral ``commas''}};
  \end{scope}

  % (b) Pathological Scan (placed with 6.5cm clearance)
  \begin{scope}[xshift=6.5cm, local bounding box=abnormalscan]
    \draw[skull] (0,0) ellipse (2.05cm and 2.25cm);
    \draw[dashed, gray!30, thin] (0,-2.0) -- (0,2.0);
    
    % Left side: Partially preserved caudate, faint putamen
    \filldraw[striatum] (-0.65,0.45) circle (0.34cm);
    \filldraw[fill=orange!45!gray!30, draw=gray!40, dashed, line width=0.6pt, rotate around={25:(-0.90,-0.2)}] (-0.90,-0.2) ellipse (0.35cm and 0.17cm);
    \filldraw[core] (-0.65,0.45) circle (0.15cm);
    
    % Right side: Complete putaminal loss ("Period" sign)
    \filldraw[striatum] (0.65,0.45) circle (0.32cm);
    \draw[draw=red!70!black, dashed, line width=0.8pt, rotate around={-25:(0.95,-0.25)}] (0.95,-0.25) ellipse (0.50cm and 0.21cm);
    \filldraw[core] (0.65,0.45) circle (0.14cm);
    
    % Deficit callout pointing rightwards, clear of all drawings
    \draw[-{Latex[length=2.0mm]}, line width=1pt, red!75!black] (1.85,-1.2) node[below, font=\scriptsize\bfseries, text=red!75!black, align=center] {Severe Putaminal Deficit\\(Flattened to ``Period'')} -- (1.05,-0.35);
    
    \node[font=\small\bfseries, align=center] at (0,-2.7) {(b) Abnormal (\texttt{is\_pathologic} = 1)\\{\footnotesize Asymmetric loss of putaminal binding}};
  \end{scope}
\end{tikzpicture}"""

(FIG_DIR / "fig_schematic.tex").write_text(FIG_SCHEMATIC, encoding="utf-8")


# -------------------------------------------------------------
# 2. Figure 2: Preprocessing Pipeline (Clean, Modular Flow)
# -------------------------------------------------------------
FIG_PREPROCESSING = r"""\begin{tikzpicture}[
  font=\footnotesize\sffamily,
  node distance=8mm and 12mm,
  stage/.style={draw=blue!40!black, rounded corners=4pt, fill=blue!5, line width=0.8pt, align=center, minimum height=1.1cm, inner sep=6pt},
  decision/.style={draw=orange!60!black, fill=orange!10, rounded corners=2pt, line width=0.8pt, align=center, minimum height=1.0cm, inner sep=5pt},
  passnode/.style={draw=teal!60!black, fill=teal!10, rounded corners=4pt, line width=0.8pt, align=center, minimum height=1.1cm, inner sep=6pt},
  failnode/.style={draw=red!60!black, fill=red!8, rounded corners=4pt, line width=0.8pt, align=center, minimum height=1.1cm, inner sep=6pt},
  arrow/.style={-{Latex[length=2.2mm]}, thick, gray!70!black},
  lbl/.style={font=\scriptsize\bfseries, text=gray!80!black}
]
  % Stage 1: Harmonization
  \node[stage, minimum width=2.8cm] (raw) {\textbf{Multi-Site NIfTI}\\10 native hospitals\\(2.46--3.9\,mm spacing)};
  \node[stage, right=of raw, minimum width=3.0cm] (headbox) {\textbf{Fixed Head Box}\\$192\times192\times144$\,mm\\2\,mm isotropic (CoM)};
  
  % Stage 2: Registration & Fallback
  \node[stage, right=of headbox, minimum width=3.0cm] (reg) {\textbf{Affine Registration}\\to label-blind\\symmetric template};
  \node[decision, below=7mm of reg, minimum width=2.8cm] (gate) {\textbf{Quality Gate}\\Transform OK?};
  
  % Paths
  \node[passnode, right=of reg, minimum width=2.6cm] (regframe) {\textbf{Registered}\\\textbf{Frame}\\(canonical)};
  \node[failnode, right=of gate, minimum width=2.6cm] (comframe) {\textbf{CoM Frame}\\(safe fallback)};
  
  % Stage 3: Normalization & Derivation
  \node[stage, right=14mm of regframe, minimum width=3.0cm] (norm) {\textbf{Ref-Region Norm.}\\Non-striatal uptake $\to 1.0$};
  \node[stage, right=of norm, minimum width=3.2cm] (inputs) {\textbf{Derived Inputs}\\$\bullet$ 3-Slab Stacks (2.5D)\\$\bullet$ Striatal Crops (3D)\\$\bullet$ Radiomics (Feat)};

  % Connecting arrows
  \draw[arrow] (raw) -- (headbox);
  \draw[arrow] (headbox) -- (reg);
  \draw[arrow] (reg) -- (gate);
  
  \draw[arrow] (reg) -- node[above, lbl] {Pass} (regframe);
  \draw[arrow] (gate) -- node[above, lbl, text=red!70!black] {Fallback} (comframe);
  
  \draw[arrow] (regframe) -- (norm);
  \draw[arrow] (comframe.east) -| ++(0.7,0) |- (norm.south west);
  \draw[arrow] (norm) -- (inputs);

\end{tikzpicture}"""

(FIG_DIR / "fig_preprocessing.tex").write_text(FIG_PREPROCESSING, encoding="utf-8")


# -------------------------------------------------------------
# 3. Figure 3: Hemisphere Arm Architecture (Compact Horizontal)
# -------------------------------------------------------------
FIG_HEMI = r"""\begin{tikzpicture}[
  font=\scriptsize\sffamily,
  scale=0.92, every node/.style={transform shape},
  box/.style={draw=blue!40!black, rounded corners=3pt, fill=blue!6, align=center, inner sep=4pt, minimum height=0.85cm},
  enc/.style={draw=teal!60!black, rounded corners=4pt, fill=teal!12, align=center, line width=0.9pt, inner sep=5pt, minimum height=1.5cm},
  head/.style={draw=orange!60!black, rounded corners=3pt, fill=orange!10, align=center, inner sep=4pt, minimum height=0.85cm},
  op/.style={draw=gray!60!black, circle, fill=gray!12, inner sep=2pt, font=\small\bfseries},
  arrow/.style={-{Latex[length=2mm]}, thick, gray!70!black}
]
  % Top row: Input & Half Crops
  \node[box, fill=purple!8, minimum width=3.8cm] (input) at (0, 3.8) {\textbf{Striatal Crop} ($48\times48\times32$)\\(Registered / CoM frame)};
  \node[box, minimum width=2.4cm] (left1) at (-2.6, 2.3) {Left Half\\$24\times48\times32$};
  \node[box, minimum width=2.4cm] (right1) at (2.6, 2.3) {Right Half\\$24\times48\times32$};
  \node[box, fill=gray!10, minimum width=2.4cm] (mirror) at (2.6, 1.0) {Midline Flip $\mathcal{M}$\\(left orientation)};

  % Mid row: Shared 3D Encoder
  \node[enc, minimum width=3.2cm] (encL) at (-2.6, -0.6) {\textbf{Shared 3D ResNet}\\\textbullet\ Conv3D Stem\\\textbullet\ 4$\times$ Block3D ($24{\to}192$)\\\textbullet\ Global Average Pool};
  \node[enc, minimum width=3.2cm] (encR) at (2.6, -0.6) {\textbf{Shared 3D ResNet}\\\textbullet\ Conv3D Stem\\\textbullet\ 4$\times$ Block3D ($24{\to}192$)\\\textbullet\ Global Average Pool};
  \node[font=\tiny\itshape, text=teal!70!black] at (0, -0.6) {Identical\\Shared Weights};
  \draw[dashed, teal!60, thick] (encL.east) -- (encR.west);

  % Feature vectors
  \node[box, fill=teal!6, minimum width=2.0cm] (fL) at (-2.6, -2.1) {$f_L \in \mathbb{R}^{192}$};
  \node[box, fill=teal!6, minimum width=2.0cm] (fR) at (2.6, -2.1) {$f_R \in \mathbb{R}^{192}$};

  % Bottom row: Dual Symmetric Heads
  \node[head, fill=green!10, minimum width=3.2cm] (combine) at (0, -3.2) {\textbf{Bilateral \& Asymmetry}\\\textbf{Feature}: $[f_L+f_R,\, |f_L-f_R|]$};
  \node[head, minimum width=2.5cm] (pairhead) at (0, -4.5) {\textbf{Pair Head MLP}\\$384 \to 128 \to 1$};
  
  \node[head, minimum width=2.0cm] (sideL) at (-4.2, -3.2) {Side Head\\(shared)};
  \node[head, minimum width=2.0cm] (sideR) at (4.2, -3.2) {Side Head\\(shared)};
  \node[op] (maxop) at (4.2, -4.5) {$\max$};

  \node[op] (sumop) at (0, -5.7) {$+$};
  \node[box, fill=orange!20, minimum width=4.0cm, line width=1pt] (output) at (0, -6.8) {\textbf{Final Logit}: $z = z_{\text{pair}} + z_{\text{side}}$\\{\scriptsize Provably invariant to left-right scan flip}};

  % Arrows
  \draw[arrow] (input.south) -| (left1.north);
  \draw[arrow] (input.south) -| (right1.north);
  \draw[arrow] (left1) -- (encL);
  \draw[arrow] (right1) -- (mirror);
  \draw[arrow] (mirror) -- (encR);
  \draw[arrow] (encL) -- (fL);
  \draw[arrow] (encR) -- (fR);
  
  \draw[arrow] (fL.south) |- (combine.west);
  \draw[arrow] (fR.south) |- (combine.east);
  \draw[arrow] (combine) -- (pairhead);
  
  \draw[arrow] (fL.west) -| (sideL.north);
  \draw[arrow] (fR.east) -| (sideR.north);
  \draw[arrow] (sideL.south) |- node[pos=0.7,above,font=\tiny]{$z_L$} (maxop.west);
  \draw[arrow] (sideR.south) -- node[right,font=\tiny]{$z_R$} (maxop.north);

  \draw[arrow] (pairhead) -- node[left,font=\tiny]{$z_{\text{pair}}$} (sumop);
  \draw[arrow] (maxop.south) |- node[pos=0.7,above,font=\tiny]{$z_{\text{side}}$} (sumop.east);
  \draw[arrow] (sumop) -- (output);
\end{tikzpicture}"""

(FIG_DIR / "fig_hemi.tex").write_text(FIG_HEMI, encoding="utf-8")


# -------------------------------------------------------------
# 4. Figure 4: Multi-Arm Ensemble Combiner
# -------------------------------------------------------------
FIG_ENSEMBLE = r"""\begin{tikzpicture}[
  font=\footnotesize\sffamily,
  scale=0.95, every node/.style={transform shape},
  arm3d/.style={draw=teal!70!black, rounded corners=3pt, minimum width=4.8cm, minimum height=0.9cm, align=left, fill=teal!10, line width=0.8pt},
  arm25d/.style={draw=blue!70!black, rounded corners=3pt, minimum width=4.8cm, minimum height=0.9cm, align=left, fill=blue!8, line width=0.8pt},
  armfeat/.style={draw=orange!70!black, rounded corners=3pt, minimum width=4.8cm, minimum height=0.9cm, align=left, fill=orange!10, line width=0.8pt},
  comb/.style={draw=green!60!black, rounded corners=4pt, minimum width=4.2cm, minimum height=3.2cm, align=center, fill=green!8, line width=0.9pt},
  outbox/.style={draw=orange!70!black, rounded corners=3pt, minimum width=2.8cm, minimum height=1.1cm, align=center, fill=orange!18, line width=0.9pt},
  arrow/.style={-{Latex[length=2.2mm]}, thick, gray!70!black}
]
  % 5 Arms
  \node[arm3d] (a1) at (0, 2.6) {\textbf{Hemisphere 3D} (HemiNet3D)\hfill\textbf{50.0\%}\\{\scriptsize Shared bilateral ResNet encoder}};
  \node[arm3d] (a2) at (0, 1.3) {\textbf{Self-Supervised 3D} (VICReg)\hfill\textbf{12.5\%}\\{\scriptsize Pretrained 3D CNN on whole brain}};
  \node[armfeat] (a3) at (0, 0.0) {\textbf{Feature Arm} (Radiomics)\hfill\textbf{12.5\%}\\{\scriptsize 116 striatal descriptors + Kernel RFF}};
  \node[arm25d] (a4) at (0, -1.3) {\textbf{Coronal 2.5D} (ConvNeXt-Tiny)\hfill\textbf{12.5\%}\\{\scriptsize 3-slab coronal projection stacks}};
  \node[arm25d] (a5) at (0, -2.6) {\textbf{Axial 2.5D} (EfficientNetV2-S)\hfill\textbf{12.5\%}\\{\scriptsize 3-slab axial projection stacks}};

  % Combiner Box
  \node[comb] (comb) at (6.2, 0.0) {
    \textbf{\large Calibration Combiner}\\[4pt]
    \parbox{3.8cm}{\centering
      \textbullet\ Log-odds conversion: $z_i = \text{logit}(p_i)$\\[2pt]
      \textbullet\ Weighted pooling: $\bar{z} = \sum w_i z_i$\\[2pt]
      \textbullet\ Temperature scaling: $T=0.90$\\[2pt]
      \textbullet\ Bayesian prior mix: $\lambda=0.01$
    }
  };

  % Output Box
  \node[outbox] (out) at (10.8, 0.0) {\textbf{Final Prediction}\\$\hat{p} \in [0, 1]$\\{\scriptsize Well-calibrated}};

  % Connections
  \foreach \a in {a1, a2, a3, a4, a5} {
    \draw[arrow] (\a.east) -- (comb.west |- \a.east);
  }
  \draw[arrow] (comb) -- (out);
\end{tikzpicture}"""

(FIG_DIR / "fig_ensemble.tex").write_text(FIG_ENSEMBLE, encoding="utf-8")


# -------------------------------------------------------------
# 5. Figure 5: Publication-Grade Shift Gap Bar Chart
# -------------------------------------------------------------
FIG_SHIFTGAP = r"""\begin{tikzpicture}[
  font=\sffamily,
  scale=1.0,
  y=22cm, x=1.6cm,
  barlabel/.style={font=\footnotesize\bfseries},
  axislabel/.style={font=\scriptsize\bfseries, text=gray!80!black}
]
  % Y Axis and gridlines
  \draw[-{Latex[length=2.2mm]}, thick, gray!70!black] (0,0) -- (0,0.30) node[above, font=\footnotesize\bfseries] {Log Loss (lower is better)};
  \draw[-{Latex[length=2.2mm]}, thick, gray!70!black] (0,0) -- (4.8,0);

  \foreach \val in {0.05, 0.10, 0.15, 0.20, 0.25} {
    \draw[gray!25, dashed] (-0.1, \val) -- (4.5, \val);
    \node[anchor=east, font=\tiny, text=gray!60!black] at (-0.12, \val) {\val};
  }

  % Bar 1: Random-Fold OOF (0.2266)
  \draw[fill=blue!45!teal, draw=blue!60!black, rounded corners=2pt, line width=0.8pt] (0.5, 0) rectangle (1.3, 0.2266);
  \node[barlabel, text=blue!60!black] at (0.9, 0.237) {0.2266};
  \node[axislabel, align=center] at (0.9, -0.025) {Random-Fold\\Nested OOF};

  % Bar 2: Leave-Site-Out OOF (0.2402)
  \draw[fill=orange!55!yellow, draw=orange!70!black, rounded corners=2pt, line width=0.8pt] (1.8, 0) rectangle (2.6, 0.2402);
  \node[barlabel, text=orange!80!black] at (2.2, 0.251) {0.2402};
  \node[axislabel, align=center] at (2.2, -0.025) {Leave-Site-Out\\(Site-Fold OOF)};

  % Bar 3: Public Leaderboard (0.2473)
  \draw[fill=red!50!orange, draw=red!70!black, rounded corners=2pt, line width=0.8pt] (3.1, 0) rectangle (3.9, 0.2473);
  \node[barlabel, text=red!80!black] at (3.5, 0.258) {\textbf{0.2473}};
  \node[axislabel, align=center] at (3.5, -0.025) {Public Leaderboard\\(\textbf{Rank $\sim$16} / 450)};

  % Shift Gap Annotation Bracket (Between Bar 1 and Bar 2)
  \draw[thick, gray!60!black] (0.9, 0.260) -- (0.9, 0.268) -- (2.2, 0.268) -- (2.2, 0.260);
  \node[font=\scriptsize\bfseries, fill=white, inner sep=2pt, text=gray!80!black] at (1.55, 0.270) {Shift Gap $\Delta = +0.0136$};

  % Public Gap Annotation Bracket (Between Bar 1 and Bar 3)
  \draw[thick, red!70!black] (0.9, 0.282) -- (0.9, 0.290) -- (3.5, 0.290) -- (3.5, 0.282);
  \node[font=\scriptsize\bfseries, fill=white, inner sep=2pt, text=red!75!black] at (2.2, 0.292) {Real Test Gap $\Delta = +0.0207$};

\end{tikzpicture}"""

(FIG_DIR / "fig_shiftgap.tex").write_text(FIG_SHIFTGAP, encoding="utf-8")


# -------------------------------------------------------------
# 6. Single-Column Shift Gap Chart for ISBI 2027 (Compact)
# -------------------------------------------------------------
FIG_SHIFTGAP_ISBI = r"""\begin{tikzpicture}[
  font=\sffamily,
  scale=0.85,
  y=17cm, x=1.2cm,
  barlabel/.style={font=\tiny\bfseries},
  axislabel/.style={font=\tiny\bfseries, text=gray!80!black}
]
  \draw[-{Latex[length=1.8mm]}, thick, gray!70!black] (0,0) -- (0,0.30) node[above, font=\tiny\bfseries] {Log loss};
  \draw[-{Latex[length=1.8mm]}, thick, gray!70!black] (0,0) -- (4.2,0);

  \foreach \val in {0.10, 0.20, 0.30} {
    \draw[gray!25, dashed] (-0.08, \val) -- (4.0, \val);
    \node[anchor=east, font=\tiny, text=gray!60!black] at (-0.1, \val) {\val};
  }

  % Bar 1
  \draw[fill=blue!45!teal, draw=blue!60!black, rounded corners=1pt] (0.4, 0) rectangle (1.1, 0.2266);
  \node[barlabel, text=blue!70!black] at (0.75, 0.237) {0.2266};
  \node[axislabel, align=center] at (0.75, -0.03) {Random\\OOF};

  % Bar 2
  \draw[fill=orange!55!yellow, draw=orange!70!black, rounded corners=1pt] (1.5, 0) rectangle (2.2, 0.2402);
  \node[barlabel, text=orange!80!black] at (1.85, 0.251) {0.2402};
  \node[axislabel, align=center] at (1.85, -0.03) {Site-Fold\\OOF};

  % Bar 3
  \draw[fill=red!50!orange, draw=red!70!black, rounded corners=1pt] (2.6, 0) rectangle (3.3, 0.2473);
  \node[barlabel, text=red!80!black] at (2.95, 0.258) {0.2473};
  \node[axislabel, align=center] at (2.95, -0.03) {Public LB\\(Rank $\sim$16)};

  % Gap bracket
  \draw[thick, gray!60!black] (0.75, 0.268) -- (0.75, 0.274) -- (1.85, 0.274) -- (1.85, 0.268);
  \node[font=\tiny\bfseries, fill=white, inner sep=1pt] at (1.3, 0.276) {$\shiftgap = 0.0136$};

  \draw[thick, red!70!black] (0.75, 0.286) -- (0.75, 0.292) -- (2.95, 0.292) -- (2.95, 0.286);
  \node[font=\tiny\bfseries, fill=white, inner sep=1pt, text=red!70!black] at (1.85, 0.294) {Gap $+0.0207$};
\end{tikzpicture}"""

(FIG_DIR / "fig_shiftgap_isbi.tex").write_text(FIG_SHIFTGAP_ISBI, encoding="utf-8")

print("Successfully generated all publication figures in paper_v2/figures/")
