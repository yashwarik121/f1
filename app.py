# app.py — F1 Dashboard (PDF + CSV only, no HTML export)
import os
import tempfile
from datetime import datetime
import streamlit as st
import pandas as pd
import fastf1
import plotly.express as px
import plotly.graph_objects as go

# try imports for PDF generation (kaleido + reportlab)
_kaleido_ok = True
_reportlab_ok = True
try:
    import kaleido  # noqa: F401
except Exception:
    _kaleido_ok = False
try:
    from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
except Exception:
    _reportlab_ok = False

# -------------------------
# CONFIG
# -------------------------
CACHE_DIR = "f1_cache"
ASSETS_DIR = "assets"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

st.set_page_config(page_title="F1 Analytics — PDF + CSV", layout="wide")
st.title("🏁 F1 WARZ")

# Team color dictionary (extend if needed)
TEAM_COLORS = {
    "Red Bull Racing": "#0600EF",
    "Mercedes": "#00D2BE",
    "Ferrari": "#DC0000",
    "McLaren": "#FF8700",
    "Alpine": "#0090FF",
    "Aston Martin": "#2B6E4A",
    "Williams": "#005AFF",
    "AlphaTauri": "#2B2B2B",
    "Alfa Romeo": "#900000",
    "Haas": "#B6B6B6",
}
FALLBACK_COLOR = "#7f7f7f"

RACE_LIST = [
    "Bahrain", "Saudi Arabian", "Australian", "Azerbaijan", "Miami",
    "Monaco", "Spanish", "Canadian", "Austrian", "British",
    "Hungarian", "Belgian", "Dutch", "Italian", "Singapore",
    "Japanese", "Qatar", "United States", "Mexico City",
    "São Paulo", "Las Vegas", "Abu Dhabi"
]
SESSION_MAP = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Qualifying": "Q", "Race": "R"}

# -------------------------
# HELPERS
# -------------------------
def team_color(team):
    if not isinstance(team, str):
        return FALLBACK_COLOR
    for k in TEAM_COLORS:
        if k.lower() in team.lower() or team.lower() in k.lower():
            return TEAM_COLORS[k]
    return FALLBACK_COLOR

def safe_load_session(year, gp, s_type):
    try:
        sess = fastf1.get_session(year, gp, s_type)
        sess.load()
        return sess, None
    except Exception as e:
        return None, str(e)

def laps_to_clean_df(laps):
    df = pd.DataFrame(laps)
    if df.empty:
        return pd.DataFrame()
    wanted = ['Driver','Team','LapNumber','LapTime','LapTimeSec',
              'Sector1Time','Sector2Time','Sector3Time',
              'Compound','Stint','PitInTime','PitOutTime','TyreLife','Position']
    available = [c for c in wanted if c in df.columns]
    if not available:
        return pd.DataFrame()
    df = df[available].copy()
    if 'LapNumber' not in df.columns:
        df['LapNumber'] = df.groupby('Driver').cumcount() + 1
    if 'LapTimeSec' not in df.columns and 'LapTime' in df.columns:
        try:
            df['LapTimeSec'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
        except Exception:
            df['LapTimeSec'] = pd.to_numeric(df['LapTime'], errors='coerce')
    for s in ['Sector1Time','Sector2Time','Sector3Time']:
        if s in df.columns:
            try:
                df[s+'Sec'] = pd.to_timedelta(df[s]).dt.total_seconds()
            except Exception:
                df[s+'Sec'] = pd.to_numeric(df[s], errors='coerce')
    if 'LapTimeSec' in df.columns:
        df = df.dropna(subset=['LapTimeSec'])
        df = df[df['LapTimeSec'] > 3]
    df['Driver'] = df['Driver'].astype(str)
    df['Team'] = df['Team'].astype(str)
    return df

# FIGURE BUILDERS (same graphs as before)
def fig_team_avg(df):
    if 'LapTimeSec' not in df.columns:
        return None
    team_avg = df.groupby('Team', as_index=False)['LapTimeSec'].mean().sort_values('LapTimeSec')
    color_map = {t: team_color(t) for t in team_avg['Team'].unique()}
    fig = px.bar(team_avg, x='Team', y='LapTimeSec',
                 title='Average Lap Time per Team (s)',
                 color='Team', color_discrete_map=color_map, text=team_avg['LapTimeSec'].round(2))
    fig.update_layout(showlegend=False, height=420)
    return fig

def fig_lap_progression(df):
    if 'LapTimeSec' not in df.columns:
        return None
    counts = df['Driver'].value_counts()
    drivers_ok = counts[counts>=3].index.tolist()
    dfp = df[df['Driver'].isin(drivers_ok)].copy()
    if dfp.empty:
        return None
    fig = go.Figure()
    driver_order = sorted(dfp['Driver'].unique(), key=lambda d: dfp[dfp['Driver']==d]['LapTimeSec'].min())
    for drv in driver_order:
        sub = dfp[dfp['Driver'] == drv].sort_values('LapNumber')
        c = team_color(sub['Team'].iloc[0]) if 'Team' in sub.columns else FALLBACK_COLOR
        fig.add_trace(go.Scatter(x=sub['LapNumber'], y=sub['LapTimeSec'], mode='lines+markers',
                                 name=drv, line=dict(color=c), marker=dict(size=4)))
    fig.update_layout(xaxis_title='Lap Number', yaxis_title='Lap Time (s)', height=450)
    return fig

def fig_stint_gantt(df):
    segs = []
    if 'Stint' not in df.columns:
        df['Stint'] = df.groupby('Driver').cumcount() + 1
    for driver, g in df.groupby('Driver'):
        for stint, sg in g.groupby('Stint'):
            start = int(sg['LapNumber'].min())
            end = int(sg['LapNumber'].max())
            comp = sg['Compound'].mode().iloc[0] if 'Compound' in sg.columns and not sg['Compound'].mode().empty else 'unknown'
            segs.append({'Driver': driver, 'Start': start, 'End': end, 'Compound': comp})
    if not segs:
        return None
    segdf = pd.DataFrame(segs)
    drivers_sorted = sorted(segdf['Driver'].unique())
    fig = go.Figure()
    comp_colors = {}
    palette = px.colors.qualitative.Dark24
    pi = 0
    for drv in drivers_sorted:
        sgs = segdf[segdf['Driver']==drv].sort_values('Start')
        for _, r in sgs.iterrows():
            comp = r['Compound']
            if comp not in comp_colors:
                comp_colors[comp] = palette[pi % len(palette)]; pi += 1
            fig.add_trace(go.Bar(x=[r['End'] - r['Start'] + 1], y=[drv], base=[r['Start']], orientation='h',
                                 marker=dict(color=comp_colors[comp]),
                                 hovertemplate=f"Driver: {drv}<br>Stint: {r['Start']}-{r['End']}<br>Compound: {comp}<extra></extra>"))
    fig.update_layout(barmode='stack', xaxis_title='Lap Number', yaxis_title='Driver', height=600, showlegend=False)
    return fig, comp_colors

def detect_pits(df):
    pits = []
    if 'PitInTime' in df.columns or 'PitOutTime' in df.columns:
        pitrows = df[df.get('PitInTime').notna() | df.get('PitOutTime').notna()]
        for _, r in pitrows.iterrows():
            pits.append({'Driver': r['Driver'], 'PitLap': int(r['LapNumber']), 'PitIn': r.get('PitInTime'), 'PitOut': r.get('PitOutTime')})
    else:
        if 'Stint' not in df.columns:
            df['Stint'] = df.groupby('Driver').cumcount() + 1
        for driver, g in df.groupby('Driver'):
            prev = None
            for _, row in g.sort_values('LapNumber').iterrows():
                if prev is None:
                    prev = row['Stint']
                elif row['Stint'] != prev:
                    pits.append({'Driver': row['Driver'], 'PitLap': int(row['LapNumber']), 'PitIn': None, 'PitOut': None})
                    prev = row['Stint']
    return pd.DataFrame(pits)

def fig_pit_timeline(pits):
    if pits.empty:
        return None
    fig = px.scatter(pits, x='PitLap', y='Driver', color='Driver', hover_data=['PitLap','PitIn','PitOut'], title='Pit Laps per Driver', height=400)
    fig.update_layout(xaxis_title='Lap Number')
    return fig

# Advanced analytics (sector, delta, positions, heatmap)
def fig_sector_breakdown(df):
    sector_cols = [c for c in df.columns if c.endswith('Sec') and 'Sector' in c]
    if not sector_cols:
        return None
    df_sect = df.groupby('Driver')[sector_cols].mean().reset_index()
    m = df_sect.melt(id_vars='Driver', value_vars=sector_cols, var_name='Sector', value_name='Seconds')
    m['Sector'] = m['Sector'].str.replace('Sec','')
    fig = px.bar(m, x='Driver', y='Seconds', color='Sector', barmode='group', title='Average Sector Times per Driver')
    return fig

def fig_delta(df, d1, d2):
    if d1 == d2:
        return None
    a = df[df['Driver']==d1][['LapNumber','LapTimeSec']].rename(columns={'LapTimeSec':'T1'})
    b = df[df['Driver']==d2][['LapNumber','LapTimeSec']].rename(columns={'LapTimeSec':'T2'})
    merged = pd.merge(a,b,on='LapNumber',how='inner')
    if merged.empty:
        return None
    merged['Delta'] = merged['T1'] - merged['T2']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged['LapNumber'], y=merged['Delta'], mode='lines+markers', name=f"{d1} - {d2} (s)"))
    fig.update_layout(title=f"Lap-by-lap Delta: {d1} minus {d2}", xaxis_title='Lap', yaxis_title='Delta (s)')
    return fig

def fig_position_change(df):
    if 'Position' not in df.columns:
        return None
    pos = df[['Driver','LapNumber','Position']].copy()
    pos = pos.dropna(subset=['Position'])
    fig = go.Figure()
    for drv in pos['Driver'].unique():
        sub = pos[pos['Driver']==drv].sort_values('LapNumber')
        fig.add_trace(go.Scatter(x=sub['LapNumber'], y=sub['Position'], mode='lines+markers', name=drv))
    fig.update_layout(title='Position by Lap', xaxis_title='Lap', yaxis_title='Position', yaxis_autorange='reversed')
    return fig

def fig_fastest_lap_heatmap(df):
    if 'LapTimeSec' not in df.columns:
        return None
    piv = df.pivot_table(index='Driver', columns='LapNumber', values='LapTimeSec', aggfunc='min')
    if piv.empty:
        return None
    piv_norm = piv.apply(lambda row: row / row.min(), axis=1)
    fig = px.imshow(piv_norm, labels=dict(x='Lap', y='Driver', color='Relative Pace'),
                    title='Fastest-lap Heatmap (normalized)')
    return fig

# EXPORT: save Plotly figs as PNG (kaleido) then make PDF (reportlab)
def save_fig_png(fig, path):
    if not _kaleido_ok:
        raise RuntimeError("Kaleido not installed. Run: python -m pip install kaleido")
    fig.write_image(path, engine='kaleido', format='png', scale=2)
    return path

def build_pdf(images_with_captions, out_path, meta=None):
    if not _reportlab_ok:
        raise RuntimeError("ReportLab not installed. Run: python -m pip install reportlab")
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []
    if meta:
        flow.append(Paragraph(meta, styles['Title']))
        flow.append(Spacer(1,12))
    for img_path, caption in images_with_captions:
        flow.append(Image(img_path, width=500, height=300))
        flow.append(Paragraph(caption, styles['Normal']))
        flow.append(Spacer(1,12))
    doc.build(flow)
    return out_path

# -------------------------
# UI
# -------------------------
st.sidebar.header("Session")
year = st.sidebar.number_input("Season", min_value=2018, max_value=2025, value=2023)
gp = st.sidebar.selectbox("Grand Prix", RACE_LIST, index=RACE_LIST.index("Monaco"))
session_label = st.sidebar.selectbox("Session", ["Practice 1","Practice 2","Practice 3","Qualifying","Race"])
session = SESSION_MAP[session_label]
load = st.sidebar.button("Load Race Data")

if load:
    with st.spinner("Loading session (may take 1-3 min first time)..."):
        sess, err = safe_load_session(year, gp, session)
    if err:
        st.error(f"Failed to load session: {err}")
    else:
        st.success(f"Loaded: {year} {gp} ({session})")
        laps = sess.laps
        df = laps_to_clean_df(laps)
        if df.empty:
            st.error("No usable lap data.")
        else:
            # top metrics
            c1,c2,c3 = st.columns(3)
            c1.metric("Teams", df['Team'].nunique())
            c2.metric("Drivers", df['Driver'].nunique())
            c3.metric("Lap records", len(df))
            st.markdown("---")

            # Team avg
            fig1 = fig_team_avg(df)
            if fig1: st.plotly_chart(fig1, use_container_width=True)
            st.markdown("---")

            # Lap progression
            fig2 = fig_lap_progression(df)
            if fig2: st.plotly_chart(fig2, use_container_width=True)
            st.markdown("---")

            # Tyre stints / Gantt
            gantt_res = fig_stint_gantt(df)
            if gantt_res:
                fig_gantt, comp_colors = gantt_res
                st.subheader("Tyre Stints")
                st.plotly_chart(fig_gantt, use_container_width=True)
                if comp_colors:
                    legend_html = "<div style='display:flex;flex-wrap:wrap;'>"
                    for comp, col in comp_colors.items():
                        legend_html += f"<div style='margin:6px;display:flex;align-items:center;'><div style='width:18px;height:12px;background:{col};margin-right:6px;border:1px solid #333'></div>{comp}</div>"
                    legend_html += "</div>"
                    st.markdown(legend_html, unsafe_allow_html=True)
            st.markdown("---")

            # Pit stops
            pits = detect_pits(df)
            if not pits.empty:
                st.subheader("Pit Stops")
                fig_pc = px.bar(pits.groupby('Driver').size().reset_index(name='PitCount').sort_values('PitCount', ascending=False),
                                x='Driver', y='PitCount', title='Pit stops per driver')
                st.plotly_chart(fig_pc, use_container_width=True)
                fig_pt = fig_pit_timeline(pits)
                if fig_pt: st.plotly_chart(fig_pt, use_container_width=True)
                st.dataframe(pits.sort_values(['Driver','PitLap']).head(200))
            else:
                st.info("No pit stop records detected.")
            st.markdown("---")

            # Advanced analytics
            st.subheader("Advanced Analytics")
            fig_sector = fig_sector_breakdown(df)
            if fig_sector: st.plotly_chart(fig_sector, use_container_width=True)
            else: st.info("Sector data not available.")

            st.markdown("Delta comparison")
            drivers = sorted(df['Driver'].unique())
            if len(drivers) >= 2:
                d1 = st.selectbox("Driver 1", drivers, index=0)
                d2 = st.selectbox("Driver 2", drivers, index=1)
                fig_delta = fig_delta(df, d1, d2)
                if fig_delta: st.plotly_chart(fig_delta, use_container_width=True)
                else: st.info("Not enough overlapping laps to compute delta.")
            else:
                st.info("Not enough drivers for delta.")

            fig_pos = fig_position_change(df)
            if fig_pos: st.plotly_chart(fig_pos, use_container_width=True)
            else: st.info("Position data not available.")

            fig_heat = fig_fastest_lap_heatmap(df)
            if fig_heat: st.plotly_chart(fig_heat, use_container_width=True)
            else: st.info("Heatmap not available.")

            st.markdown("---")

            # CSV download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download cleaned lap CSV", csv, file_name=f"{gp}_{year}_{session}_laps.csv", mime='text/csv')

            # PDF export (only PDF now)
            if st.button("Download PDF Report (charts + meta)"):
                if not _kaleido_ok or not _reportlab_ok:
                    missing = []
                    if not _kaleido_ok: missing.append("kaleido")
                    if not _reportlab_ok: missing.append("reportlab")
                    st.error(f"Cannot create PDF — missing packages: {', '.join(missing)}. Install with:\npython -m pip install {' '.join(missing)}")
                else:
                    with st.spinner("Generating PDF..."):
                        try:
                            figs_for_pdf = []
                            captions = []

                            # same ordering as displayed
                            if fig1:
                                figs_for_pdf.append(('Team average', fig1))
                                captions.append('Average lap time per team')
                            if fig2:
                                figs_for_pdf.append(('Lap progression', fig2))
                                captions.append('Lap time progression for drivers')
                            if gantt_res:
                                figs_for_pdf.append(('Tyre stints', fig_gantt))
                                captions.append('Tyre stints (Gantt-style)')
                            if not pits.empty and fig_pt:
                                figs_for_pdf.append(('Pit timeline', fig_pt))
                                captions.append('Pit stop timeline')
                            if fig_sector:
                                figs_for_pdf.append(('Sector breakdown', fig_sector))
                                captions.append('Average sector times per driver')
                            if fig_delta:
                                figs_for_pdf.append(('Delta', fig_delta))
                                captions.append(f'Lap-by-lap delta: {d1} minus {d2}')
                            if fig_pos:
                                figs_for_pdf.append(('Positions', fig_pos))
                                captions.append('Position by lap')
                            if fig_heat:
                                figs_for_pdf.append(('Heatmap', fig_heat))
                                captions.append('Normalized fastest-lap heatmap')

                            tmp_imgs = []
                            for (name, fig), cap in zip(figs_for_pdf, captions):
                                tmpf = os.path.join(tempfile.gettempdir(), f"f1_{name.replace(' ','_')}.png")
                                save_fig_png = fig.write_image  # use plotly write_image (kaleido)
                                fig.write_image(tmpf, engine='kaleido', format='png', scale=2)
                                tmp_imgs.append((tmpf, cap))

                            # Build PDF
                            pdf_path = os.path.join(tempfile.gettempdir(), f"f1_report_{gp}_{year}_{session}.pdf")
                            meta = f"{gp} — {year} ({session})    Generated: {datetime.utcnow().isoformat()} UTC"
                            # use reportlab to stitch images
                            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                            styles = getSampleStyleSheet()
                            flow = []
                            flow.append(Paragraph(meta, styles['Title']))
                            flow.append(Spacer(1,12))
                            for imgp, cap in tmp_imgs:
                                flow.append(Image(imgp, width=500, height=300))
                                flow.append(Paragraph(cap, styles['Normal']))
                                flow.append(Spacer(1,12))
                            doc.build(flow)

                            with open(pdf_path, 'rb') as f:
                                st.download_button("Download PDF", f, file_name=os.path.basename(pdf_path))
                        except Exception as e:
                            st.error(f"PDF generation failed: {e}")

            st.success("Done — CSV ready. Use PDF export for a full report (requires kaleido + reportlab).")
