"""
make_significance_table.py
======================================================================
The appendix tables, generated from paper_tests.py's output so they cannot drift
from it.

    python paper_tests.py && python make_significance_table.py

Writes, into output/stats/:
    significance_table.tex   Table A1, the four test families   (\\input-ready)
    effective_n_table.tex    Table A2, ICC -> design effect -> effective n
    significance_table.md    both, as plain text for reading in a terminal

WHY TWO TABLES AND NOT ONE
---------------------------
A1 and A2 answer different questions. A1 asks "is this difference real"; A2 asks
"how much evidence is 100 runs actually worth in this cell". A2 is not a list of
tests and putting it in the same table would imply it is.

THE n COLUMN IS NOT ONE KIND OF NUMBER, AND SAYS SO
----------------------------------------------------
Three of the four families are computed on RUNS (300 for a log-rank over three
conditions, 200 for a two-group Fisher or Brown-Forsythe) and one on MAP SEEDS
(10, paired). That is not an inconsistency to hide: the run-level families are
testing things that only exist at run level — when an individual run crossed
viability, whether it collapsed, how far runs scatter within a condition — and
aggregating them to seed means would delete the quantity under test. Table A2 is
the standing caveat on all three, which is why it sits directly underneath.
"""

import argparse
import os

import numpy as np
import pandas as pd

BS2 = chr(92) * 2          # a literal LaTeX row terminator
ENV = {'baseline': 'Baseline', 'hard': 'Predator'}
COND = {'evolution': 'Evolution', 'learning': 'Learning',
        'pure_rl': 'Pure RL'}

# Order, and the caption sentence, for each family block in Table A1.
FAMILIES = [
    ('Log-rank, time to sustained viability',
     r'Generations until mean fitness sits at or above the viability threshold '
     r'of 4.375 for ten consecutive generations (Figure~4). Runs that never '
     r'cross are censored at generation 1000. The stratified test compares runs '
     r'only against other runs on the same terrain and is the one we quote; the '
     r'unstratified statistic is given beside it so the effect of the clustering '
     r'is visible. Both were implemented directly and verified against '
     r'\texttt{lifelines} in the unstratified case.'),
    ("Fisher's exact, runs below viability",
     r'Number of runs whose converged fitness fell below 4.375, predator '
     r'environment only --- no run in either condition fell below it in the '
     r'baseline (Figure~2). Fisher is exact, so the zero cell is not a problem; '
     r'a $\chi^2$ approximation here would not be trustworthy. Holm-adjusted '
     r'over the two contrasts.'),
    ('Brown-Forsythe, equality of variance',
     r'Spread of converged fitness across the 100 runs of each condition '
     r"(Figure~2, Table~4). Brown--Forsythe is Levene's test centred on the "
     r'median, which is what makes it survive the bimodal distributions the '
     r'predator environment produces. Pairwise contrasts are Holm-adjusted over '
     r'the two.'),
    ('Wilcoxon signed-rank, inherited genomes',
     r'Inherited genomes re-evaluated with learning disabled, paired across the '
     r'ten map seeds at each founder-dump checkpoint (Figure~3). Positive '
     r'favours the learning condition. Rank-based because $n=10$ is too small to '
     r'assume normality; Holm-adjusted over the four checkpoints within each '
     r'environment.'),
]


def _p(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return float(v)


def _p_tex(v):
    v = _p(v)
    if v is None:
        return '--'
    if v < 0.001:
        m, e = f'{v:.0e}'.split('e')
        return rf'${m}{{\times}}10^{{{int(e)}}}$'
    return f'{v:.3f}'


def _p_md(v):
    v = _p(v)
    if v is None:
        return '--'
    return (f'{v:.0e}'.replace('e-0', 'e-')) if v < 0.001 else f'{v:.3f}'


def _stat_tex(r):
    if not np.isfinite(r.statistic):
        return r'$\infty$'
    if r.statistic_name == 'odds ratio' and r.statistic > 1e6:
        return r'$\infty$'
    if r.statistic_name.startswith('mean paired'):
        return f'{r.statistic:+.3f}'
    return f'{r.statistic:.4g}'


A1_HEAD = r"""\begin{table*}[t]
  \centering
  \caption{The statistical tests reported in Section~\ref{sec:results}, one test
  per claim. Each block names its own statistic, because the families answer
  different questions and their statistics are not comparable to one another.
  The $n$ column gives the unit each test is computed on: the first three
  families are computed on individual runs, because when a run crossed
  viability, whether it collapsed, and how far runs scatter are quantities that
  exist only at run level; the fourth is paired across the ten map seeds.
  Runs sharing a map seed share their terrain and are not independent, so the
  run-level $n$ overstates the evidence --- Table~\ref{tab:neff} quantifies by
  how much in each cell. $\alpha = 0.05$ throughout.}
  \label{tab:significance}
  \small
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{@{}llrrrr@{}}
    \toprule
    Contrast & Detail & Statistic & $n$ & $p$ & $p$ (Holm) \\
"""

A2 = r"""\begin{table}[t]
  \centering
  \caption{Effective sample size per cell. Runs sharing a map seed are not
  independent, so the 100 runs in a cell are worth fewer than 100 independent
  observations. The design effect is $1 + (m-1)\,\mathrm{ICC}$ with $m = 10$
  runs per seed, and $n_{\mathrm{eff}} = 100 / \mathrm{deff}$. The ICC is the
  one reported in Table~\ref{tab:icc}. The spread is the point: baseline
  evolution carries about as much evidence as %d independent runs, predator
  pure RL about %d.}
  \label{tab:neff}
  \small
  \setlength{\tabcolsep}{6pt}
  \begin{tabular}{@{}llrrr@{}}
    \toprule
    Environment & Condition & ICC & Design effect & $n_{\mathrm{eff}}$ \\
    \midrule
%s    \bottomrule
  \end{tabular}
\end{table}
"""

FOOT = """    \\bottomrule
  \\end{tabular}
\\end{table*}
"""


def a1_tex(df):
    out = [A1_HEAD]
    for fam, caption in FAMILIES:
        g = df[df.family == fam]
        if g.empty:
            continue
        out.append('    \\midrule\n')
        out.append('    \\multicolumn{6}{@{}p{\\textwidth}@{}}{\\footnotesize '
                   '\\textit{' + caption + '}}\\\\[2pt]\n')
        for _i, r in g.iterrows():
            out.append(f'    {ENV[r.environment]}: {r.contrast} & {r.detail} & '
                       f'{_stat_tex(r)} & {int(r.n)} & {_p_tex(r.p)} & '
                       f'{_p_tex(r.get("p_holm"))} \\\\\n')
    out.append(FOOT)
    return ''.join(out)


def a2_tex(eff):
    body = ''.join(
        f'    {ENV[r.environment]} & {COND[r.condition]} & '
        f'{r.icc:.3f} & {r.design_effect:.2f} & {r.n_effective:.0f} \\\\\n'
        for _i, r in eff.iterrows())
    lo = eff.n_effective.min()
    hi = eff.n_effective.max()
    return A2 % (round(lo), round(hi), body)


def to_md(df, eff):
    out = ['## Table A1 — tests reported in Section 5\n']
    for fam, caption in FAMILIES:
        g = df[df.family == fam]
        if g.empty:
            continue
        plain = (caption.replace('~', ' ').replace(r'\texttt', '')
                 .replace(r'$\chi^2$', 'chi-square').replace('---', '—')
                 .replace(r'\,', ' ').replace('$', '').replace('{', '')
                 .replace('}', '').replace('\\', ''))
        out.append(f'\n**{fam}.** {plain}\n')
        out.append('| Contrast | Detail | Statistic | n | p | p (Holm) |')
        out.append('|---|---|---:|---:|---:|---:|')
        for _i, r in g.iterrows():
            s = _stat_tex(r).replace(r'$\infty$', 'inf').replace('$', '')
            out.append(f'| {ENV[r.environment]}: {r.contrast} | {r.detail} | '
                       f'{s} | {int(r.n)} | {_p_md(r.p)} | '
                       f'{_p_md(r.get("p_holm"))} |')
    out.append('\n## Table A2 — effective sample size\n')
    out.append('| Environment | Condition | ICC | Design effect | n_eff (of 100) |')
    out.append('|---|---|---:|---:|---:|')
    for _i, r in eff.iterrows():
        out.append(f'| {ENV[r.environment]} | {COND[r.condition]} '
                   f'| {r.icc:.3f} | {r.design_effect:.2f} | {r.n_effective:.0f} |')
    return '\n'.join(out) + '\n'


# The six rows the MAIN TEXT quotes: the omnibus per environment for each
# family, plus the two Fisher contrasts. Everything else stays in the appendix.
HEADLINE = [
    ('baseline', 'Log-rank, time to sustained viability',
     'A vs B vs C, stratified by map seed', 'A vs B vs C, time to viability'),
    ('baseline', 'Brown-Forsythe, equality of variance',
     'A vs B vs C (omnibus)', 'A vs B vs C, converged fitness variance'),
    ('hard', 'Log-rank, time to sustained viability',
     'A vs B vs C, stratified by map seed', 'A vs B vs C, time to viability'),
    ('hard', "Fisher's exact, runs below viability",
     'A vs B', 'A vs B, runs below threshold'),
    ('hard', "Fisher's exact, runs below viability",
     'A vs C', 'A vs C, runs below threshold'),
    ('hard', 'Brown-Forsythe, equality of variance',
     'A vs B vs C (omnibus)', 'A vs B vs C, converged fitness variance'),
]
# The Wilcoxon family has four checkpoints per environment and the main table has
# room for one row each, so the row shown is chosen, not fixed: the EARLIEST
# checkpoint that clears alpha after Holm, or — when none does — the checkpoint
# with the smallest adjusted p, labelled so that the null reads as a null rather
# than as a positive result that happened to be quoted.
WILCOXON_FAMILY = 'Wilcoxon signed-rank, inherited genomes'


def wilcoxon_rows(df, env, mode='all'):
    """The Wilcoxon checkpoint rows to show for one environment.

    mode='all'        every founder-dump checkpoint (the default): the point of
                      the family is that the gap OPENS and then CLOSES, and a
                      single row cannot show a gap closing.
    mode='first-last' the earliest and latest checkpoint only, when space is
                      tight — still enough to show the convergence.
    mode='auto'       one row: the earliest checkpoint clearing alpha, or the
                      one that came closest when none does.
    """
    g = df[(df.family == WILCOXON_FAMILY) & (df.environment == env)].copy()
    if g.empty:
        return []
    g['gen'] = g.contrast.str.extract(r'(\d+)$').astype(int)
    g = g.sort_values('gen')
    if mode == 'auto':
        sig = g[g.p_holm < 0.05]
        r = sig.iloc[0] if len(sig) else g.loc[g.p_holm.idxmin()]
        tag = ('first significant at' if len(sig)
               else 'no checkpoint significant; closest at')
        return [(r, f'B vs A, inherited genome fitness, {tag} '
                    f'generation {int(r.gen)}')]
    keep = g if mode == 'all' else g.iloc[[0, -1]]
    return [(r, f'B vs A, inherited genome fitness at generation {int(r.gen)}')
            for _i, r in keep.iterrows()]


SHORT_TEST = {'Log-rank, time to sustained viability': 'Log-rank (seed-stratified)',
              "Fisher's exact, runs below viability": "Fisher's exact",
              'Brown-Forsythe, equality of variance': 'Brown--Forsythe',
              WILCOXON_FAMILY: 'Wilcoxon signed-rank (paired by seed)'}

STAT_SYM = {'chi-square': r'$\chi^2(2)$', 'odds ratio': 'OR', 'W': '$W$'}

HEADLINE_HEAD = r"""\begin{table}[t]
  \centering
  \caption{The tests behind the four claims of Section~\ref{sec:results}.
  Conditions are A (evolution), B (learning) and C (pure RL). The log-rank and
  Brown--Forsythe rows are omnibus tests across all three conditions, so each is
  a single test and needs no adjustment; the two Fisher rows form one family of
  two and carry a Holm-adjusted $p$ alongside the raw one. Each Wilcoxon row
  is one founder-dump checkpoint, Holm-adjusted over the four checkpoints in
  that environment; the gap opening and then closing is the assimilation
  signature and is why the checkpoints are shown rather than summarised. An
  infinite odds
  ratio means the learning condition had no runs below the threshold. Per-pair
  contrasts and the inherited-genome checkpoints are in
  Table~\ref{tab:significance}, and Table~\ref{tab:neff} gives the effective
  sample size behind each run-level $n$.}
  \label{tab:headline}
  \small
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{@{}lllrrr@{}}
    \toprule
    Environment & Comparison & Test & Statistic & $p$ & $p$ (Holm) \\
    \midrule
"""


WILCOXON_HEAD = r"""\begin{table}[t]
  \centering
  \caption{Inherited-genome fitness, condition B against condition A, at each
  founder-dump checkpoint. Each row is a Wilcoxon signed-rank test on the ten
  paired map-seed means, so the two conditions are always compared on identical
  terrain. $W$ is the smaller of the two signed-rank sums, which always total
  55 at $n=10$; small $W$ therefore means the larger differences fall
  consistently on one side. $W$ is unsigned, so the direction is given by the
  last column --- in the baseline the early checkpoints favour condition~A,
  whereas in the predator environment every checkpoint favours condition~B.
  $p$ is exact, enumerated over all $2^{10}$ sign assignments rather than
  approximated, and is Holm-adjusted over the four checkpoints within each
  environment. At $n=10$ the rejection region is $W \le 8$ and the smallest
  attainable $p$ is 0.002.}
  \label{tab:wilcoxon}
  \small
  \setlength{\tabcolsep}{6pt}
  \begin{tabular}{@{}lrrrr@{}}
    \toprule
    Generation & $W$ & $p$ & $p$ (Holm) & Seeds favouring B \\
"""


def wilcoxon_table_tex(df):
    out = [WILCOXON_HEAD]
    for env in ('baseline', 'hard'):
        g = df[(df.family == WILCOXON_FAMILY) & (df.environment == env)].copy()
        if g.empty:
            continue
        g['gen'] = g.contrast.str.extract(r'(\d+)$').astype(int)
        out.append('    ' + BS2[0] + 'midrule' + chr(10))
        out.append(f'    {BS2[0]}multicolumn{{5}}{{@{{}}l@{{}}}}'
                   f'{{{BS2[0]}textit{{{ENV[env]} environment}}}} '
                   + BS2 + chr(10))
        for _i, r in g.sort_values('gen').iterrows():
            out.append(f'    {int(r.gen)} & {r.statistic:.0f} & {_p_tex(r.p)} & '
                       f'{_p_tex(r.p_holm)} & {int(r.n_favouring_b)} of 10 '
                       + BS2 + chr(10))
    out.append('    ' + BS2[0] + 'bottomrule' + chr(10)
               + '  ' + BS2[0] + 'end{tabular}' + chr(10)
               + BS2[0] + 'end{table}' + chr(10))
    return ''.join(out)


def wilcoxon_table_md(df):
    out = ['## Wilcoxon signed-rank, inherited genomes (B vs A)\n']
    for env in ('baseline', 'hard'):
        g = df[(df.family == WILCOXON_FAMILY) & (df.environment == env)].copy()
        if g.empty:
            continue
        g['gen'] = g.contrast.str.extract(r'(\d+)$').astype(int)
        out.append(f'\n**{ENV[env]} environment**\n')
        out.append('| Generation | W | p | p (Holm) | Seeds favouring B | '
                   'Mean difference |')
        out.append('|---:|---:|---:|---:|---:|---:|')
        for _i, r in g.sort_values('gen').iterrows():
            out.append(f'| {int(r.gen)} | {r.statistic:.0f} | {_p_md(r.p)} | '
                       f'{_p_md(r.p_holm)} | {int(r.n_favouring_b)} of 10 | '
                       f'{r.mean_difference:+.2f} |')
    return chr(10).join(out) + chr(10)


def _headline_rows(df, mode='all'):
    """HEADLINE, with each environment's Wilcoxon rows appended to its block."""
    rows = list(HEADLINE)
    out = []
    for i, entry in enumerate(rows):
        out.append(entry)
        env = entry[0]
        last_of_env = (i + 1 == len(rows)) or rows[i + 1][0] != env
        if last_of_env:
            for r, label in wilcoxon_rows(df, env, mode):
                out.append((env, WILCOXON_FAMILY, None, (r, label)))
    return out


def headline_tex(df, mode='all'):
    out = [HEADLINE_HEAD]
    for env, fam, contrast, label in _headline_rows(df, mode):
        r = df[(df.environment == env) & (df.family == fam)
               & (df.contrast == contrast)].iloc[0] if contrast else label[1]
        if contrast:
            text = label
        else:
            r, text = label
        stat = _stat_tex(r)
        out.append(f'    {ENV[env]} & {text} & {SHORT_TEST[fam]} & '
                   f'{STAT_SYM[r.statistic_name]} = {stat} & {_p_tex(r.p)} & '
                   f'{_p_tex(r.get("p_holm"))} ' + BS2 + '\n')
    out.append(FOOT.replace('table*', 'table'))
    return ''.join(out)


def headline_md(df, mode='all'):
    out = ['## Main-text table — the tests behind the four claims\n',
           '| Environment | Comparison | Test | Statistic | p | p (Holm) |',
           '|---|---|---|---:|---:|---:|']
    for env, fam, contrast, label in _headline_rows(df, mode):
        if contrast:
            r = df[(df.environment == env) & (df.family == fam)
                   & (df.contrast == contrast)].iloc[0]
            text = label
        else:
            r, text = label
        stat = _stat_tex(r).replace(r'$\infty$', 'inf').replace('$', '')
        sym = STAT_SYM[r.statistic_name].replace('$', '').replace(r'\chi', 'chi')
        out.append(f'| {ENV[env]} | {text} | {SHORT_TEST[fam].replace("--", "-")} '
                   f'| {sym} = {stat} | {_p_md(r.p)} | {_p_md(r.get("p_holm"))} |')
    return chr(10).join(out) + chr(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stats', default=os.path.join('output', 'stats'))
    ap.add_argument('--checkpoints', choices=('all', 'first-last', 'auto'),
                    default='all',
                    help='how many inherited-genome checkpoints the MAIN table '
                         'shows per environment (default all four; Table A1 '
                         'always shows all of them)')
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.stats, 'paper_tests.csv'))
    eff = pd.read_csv(os.path.join(args.stats, 'effective_n.csv'))

    for name, text in (('wilcoxon_table.tex', wilcoxon_table_tex(df)),
                       ('wilcoxon_table.md', wilcoxon_table_md(df)),
                       ('headline_table.tex', headline_tex(df, args.checkpoints)),
                       ('headline_table.md', headline_md(df, args.checkpoints)),
                       ('significance_table.tex', a1_tex(df)),
                       ('effective_n_table.tex', a2_tex(eff)),
                       ('significance_table.md', to_md(df, eff))):
        with open(os.path.join(args.stats, name), 'w') as f:
            f.write(text)
        print(f'wrote {os.path.join(args.stats, name)}')
    print(f'  Table A1: {len(df)} tests in {df.family.nunique()} families')
    print(f'  Table A2: {len(eff)} cells')


if __name__ == '__main__':
    main()
