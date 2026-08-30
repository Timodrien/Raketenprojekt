import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from scipy.interpolate import interp1d

# ============================================================
# KONSTANTEN & TREIBSTOFFDATEN
# ============================================================
G0 = 9.81

PROPELLANTS = {
    "KNSB 65/35": {
        "density": 1750, "a": 4.58e-5, "n": 0.319,
        "cstar": 890, "cf": 1.50, "isp": 150,
    },
    "KNSB+Al 65/30/5": {
        "density": 1800, "a": 5.0e-5, "n": 0.320,
        "cstar": 940, "cf": 1.55, "isp": 160,
    }
}

MOTOR_CLASSES = [
    (2.5,'A'),(5,'B'),(10,'C'),(20,'D'),(40,'E'),(80,'F'),
    (160,'G'),(320,'H'),(640,'I'),(1280,'J'),(2560,'K'),
    (5120,'L'),(10240,'M'),(20480,'N'),(40960,'O')
]

thrust_data   = {}   # Motor Design → Flugrechner Pipeline
thrust_data_s2 = {}  # Stufe 2 Motor

# ============================================================
# PHYSIK
# ============================================================
def air_density(h):
    return 1.225 * np.exp(-h / 8500)

def speed_of_sound(h):
    T = max(288.15 - 0.0065 * min(h, 11000), 216.65)
    return np.sqrt(1.4 * 287 * T)

def motor_class(imp):
    for lim, let in MOTOR_CLASSES:
        if imp <= lim: return let
    return "P+"

def simulate_motor(Do, Di_init, Lg, N_seg, Dt, props, dt=0.001):
    rho, a, n, cstar, cf = (props["density"], props["a"], props["n"],
                             props["cstar"], props["cf"])
    At  = np.pi*(Dt/2)**2
    Vp0 = N_seg * np.pi/4 * (Do**2 - Di_init**2) * Lg
    mp0 = rho * Vp0
    times, thrusts, pmasses = [0.], [0.], [mp0]
    Dc, mp, t = Di_init, mp0, 0.
    while Dc < Do and mp > 1e-6:
        Ab = N_seg*(np.pi*Dc*Lg + 2*np.pi/4*(Do**2-Dc**2))
        try:
            P = (rho*a*Ab*cstar/At)**(1./(1.-n))
        except:
            break
        r = a*P**n;  F = cf*At*P
        t += dt
        times.append(t); thrusts.append(F)
        Dc += 2*r*dt
        mp = max(mp - rho*Ab*r*dt, 0.)
        pmasses.append(mp)
    times.append(t+dt); thrusts.append(0.); pmasses.append(0.)
    return (np.array(times), np.array(thrusts),
            np.array(pmasses), mp0)

def cd_model(mach, Cd_base):
    """Realistisches Cd-Modell mit transsonischem Anstieg."""
    if mach < 0.8:
        return Cd_base
    elif mach < 1.0:
        return Cd_base * (1 + 3.0*(mach-0.8))
    elif mach < 1.5:
        return Cd_base * (1.6 - 0.4*(mach-1.0))
    else:
        return Cd_base * 1.4

def simulate_flight(t_thr, F_thr, t_mp, prop_m, mp0,
                    m_struct, diam, h0=0., Cd=0.35, dt=0.05):
    thrust_fn = interp1d(t_thr, F_thr, bounds_error=False, fill_value=0.)
    mass_fn   = interp1d(t_mp,  prop_m, bounds_error=False, fill_value=0.)
    A = np.pi*(diam/2)**2
    burn_end = t_thr[-1]
    v, h = 0., h0
    times, heights, velocities = [0.], [h0], [0.]
    t = 0.
    while True:
        t += dt
        mp_now = float(mass_fn(t))
        mass   = max(m_struct + mp_now, m_struct)
        F_t    = float(thrust_fn(t))
        rho    = air_density(h)
        sos    = speed_of_sound(h)
        mach   = abs(v)/sos if sos > 0 else 0
        cd     = cd_model(mach, Cd)
        F_d    = 0.5*cd*A*rho*v*abs(v)
        acc    = (F_t - F_d - mass*G0)/mass
        v     += acc*dt
        h      = max(h + v*dt, 0.)
        times.append(t); heights.append(h); velocities.append(v)
        if v <= 0 and t > burn_end+5 and h < 1.: break
        if t > 2000.: break
    return np.array(times), np.array(heights), np.array(velocities)

def simulate_2stage(td1, td2, ms1, diam1, ms2, diam2,
                    coast_s=0.5, Cd=0.35, dt=0.05):
    t1,F1,pm1,mp01 = td1["times"],td1["thrusts"],td1["prop_masses"],td1["mp0"]
    t2,F2,pm2,mp02 = td2["times"],td2["thrusts"],td2["prop_masses"],td2["mp0"]
    tf1  = interp1d(t1, F1, bounds_error=False, fill_value=0.)
    mf1  = interp1d(t1, pm1, bounds_error=False, fill_value=0.)
    sep_t = t1[-1] + coast_s
    tf2  = interp1d(t2+sep_t, F2, bounds_error=False, fill_value=0.)
    mf2  = interp1d(t2+sep_t, pm2, bounds_error=False, fill_value=0.)
    burn2_end = sep_t + t2[-1]
    v, h, t, stage = 0., 0., 0., 1
    h_sep = v_sep = 0.
    times_o, heights_o, vels_o = [], [], []
    while True:
        t += dt
        if t < sep_t:
            mp_n = float(mf1(t))
            mass = ms1 + mp_n + ms2 + mp02
            diam = diam1
            Ft   = float(tf1(t))
        else:
            if stage == 1:
                h_sep, v_sep, stage = h, v, 2
            mp_n = float(mf2(t))
            mass = max(ms2 + mp_n, ms2)
            diam = diam2
            Ft   = float(tf2(t))
        A   = np.pi*(diam/2)**2
        rho = air_density(h)
        sos = speed_of_sound(h)
        mach = abs(v)/sos if sos > 0 else 0
        cd   = cd_model(mach, Cd)
        Fd   = 0.5*cd*A*rho*v*abs(v)
        acc  = (Ft - Fd - mass*G0)/mass
        v   += acc*dt
        h    = max(h + v*dt, 0.)
        times_o.append(t); heights_o.append(h); vels_o.append(v)
        if v <= 0 and t > burn2_end+5 and h < 1.: break
        if t > 3000.: break
    return (np.array(times_o), np.array(heights_o), np.array(vels_o),
            h_sep, v_sep)

# ============================================================
# NOSECONE-AERODYNAMIK
# ============================================================
def cd_nosecone(shape, fineness, calibers):
    """Vereinfachte Cd-Schätzung basierend auf Nosecone-Geometrie.
    Fineness = L/D des Nosecone, calibers = Rumpflänge/Durchmesser"""
    base = {
        "Kegelförmig":      0.55,
        "Ogive (klassisch)":0.38,
        "Von Kármán Ogive": 0.28,
        "Parabolisch":      0.32,
        "Elliptisch":       0.42,
        "Halbkugel":        0.65,
    }.get(shape, 0.40)
    # Korrekturen
    fn_corr   = max(0.7, 1.0 - 0.06*(fineness-3))
    body_corr = max(0.85, 1.0 - 0.005*(calibers-10))
    return round(base * fn_corr * body_corr, 3)

# ============================================================
# STABILITÄTSRECHNER (Barrowman-Methode, vereinfacht)
# ============================================================
def barrowman_stability(d_body, d_nose, l_nose, n_fins,
                        s_fin, cr_fin, ct_fin, xt_fin,
                        l_body):
    """
    Vereinfachter Barrowman-Ansatz für Druckpunkt-Berechnung.
    Gibt zurück: CP-Position von Raketennase (m), CN_gesamt
    """
    # Nosecone-Beitrag (Kegelförmig/Ogive näherung)
    CN_nose = 2.0
    X_nose  = 0.466 * l_nose

    # Fin-Beitrag (4 trapezförmige Fins)
    r  = d_body / 2
    s_ = s_fin
    lm = np.sqrt((cr_fin - ct_fin)**2 / 4 + s_fin**2)
    tf = cr_fin + ct_fin   # trapez sehne
    kf = (1 + r/(s_+r))

    CN_fin = (4*n_fins*(s_/d_body)**2) / (1 + np.sqrt(1+(2*lm/tf)**2))
    CN_fin *= kf

    X_fin  = (l_body - xt_fin) - (lm*(cr_fin+2*ct_fin))/(3*tf) + (tf/6)*(1+(cr_fin+ct_fin)/(cr_fin))

    # Gesamt-CP
    CN_total = CN_nose + CN_fin
    if CN_total > 0:
        X_cp = (CN_nose*X_nose + CN_fin*X_fin) / CN_total
    else:
        X_cp = l_nose

    return X_cp, CN_total

# ============================================================
# HAUPTFENSTER
# ============================================================
root = tk.Tk()
root.title("Weide Robotics – Raketensimulation v3.0")
sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry(f"{sw}x{sh}")

nb = ttk.Notebook(root)
nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

def lbl(parent, text, **kw):
    return ttk.Label(parent, text=text, **kw)

def entry(parent, default, width=16):
    e = ttk.Entry(parent, width=width)
    e.insert(0, default)
    return e

def section(parent, text):
    f = ttk.LabelFrame(parent, text=text)
    return f

# ============================================================
# TAB 1: MOTOR DESIGN
# ============================================================
tab_motor = ttk.Frame(nb)
nb.add(tab_motor, text="  Motor Design  ")

ml = section(tab_motor, "BATES Grain Parameter")
ml.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8, ipadx=4, ipady=4)

motor_f = [
    ("Außendurchmesser Do (mm):", "60"),
    ("Kerndurchmesser Di (mm):",  "24"),
    ("Segmentlänge Lg (mm):",     "200"),
    ("Anzahl Segmente:",          "3"),
    ("Düsenhals Dt (mm):",        "14"),
]
motor_entries = {}
for lbl_t, default in motor_f:
    lbl(ml, lbl_t).pack(anchor=tk.W, padx=6, pady=(5,0))
    e = entry(ml, default)
    e.pack(padx=6, fill=tk.X)
    motor_entries[lbl_t] = e

lbl(ml, "Treibstoff:").pack(anchor=tk.W, padx=6, pady=(5,0))
prop_var = tk.StringVar(value="KNSB 65/35")
ttk.OptionMenu(ml, prop_var, "KNSB 65/35", *PROPELLANTS.keys()).pack(padx=6, fill=tk.X)

ttk.Separator(ml).pack(fill=tk.X, padx=6, pady=8)
motor_res = lbl(ml, "", justify=tk.LEFT)
motor_res.pack(padx=6)

mr = ttk.Frame(tab_motor)
mr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=8)
fig_m, (ax_thr, ax_prs) = plt.subplots(2,1, figsize=(8,6))
fig_m.tight_layout(pad=3)
canvas_m = FigureCanvasTkAgg(fig_m, master=mr)
canvas_m.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def run_motor(store=thrust_data, result_lbl=None):
    vals  = [e.get() for e in motor_entries.values()]
    Do,Di,Lg = float(vals[0])/1000, float(vals[1])/1000, float(vals[2])/1000
    N, Dt = int(float(vals[3])), float(vals[4])/1000
    props = PROPELLANTS[prop_var.get()]
    times, thrusts, pmasses, mp0 = simulate_motor(Do,Di,Lg,N,Dt,props)
    store.update({"times":times,"thrusts":thrusts,"prop_masses":pmasses,
                  "mp0":mp0,"Do":Do,"props":props})
    Fmax = thrusts.max()
    Favg = np.mean(thrusts[thrusts>1]) if np.any(thrusts>1) else 0
    imp  = np.trapezoid(thrusts,times)
    txt = (f"Max Schub:    {Fmax:.0f} N\n"
           f"Avg Schub:    {Favg:.0f} N\n"
           f"Brenndauer:   {times[-1]:.3f} s\n"
           f"Gesamtimpuls: {imp:.0f} Ns\n"
           f"Treibstoff:   {mp0*1000:.0f} g\n"
           f"Motorklasse:  {motor_class(imp)}\n\n"
           f"→ Flugrechner oder Stufe 1!")
    if result_lbl: result_lbl.config(text=txt)
    else:          motor_res.config(text=txt)
    # Plots
    ax_thr.clear(); ax_thr.plot(times, thrusts, color="#e63946", lw=2)
    ax_thr.fill_between(times, thrusts, alpha=0.15, color="#e63946")
    ax_thr.set(xlabel="Zeit (s)", ylabel="Schub (N)", title="Schubkurve")
    ax_thr.grid(True, alpha=0.4); ax_thr.set_ylim(bottom=0)
    ax_prs.clear()
    props_data = PROPELLANTS[prop_var.get()]
    pressures = []
    for i,F in enumerate(thrusts):
        At = np.pi*(Dt/2)**2
        if F > 0:
            pressures.append(F/(props_data["cf"]*At)/1e6)
        else:
            pressures.append(0)
    ax_prs.plot(times, pressures, color="#f4a261", lw=2)
    ax_prs.fill_between(times, pressures, alpha=0.15, color="#f4a261")
    ax_prs.set(xlabel="Zeit (s)", ylabel="Kammerdruck (MPa)", title="Druckverlauf")
    ax_prs.grid(True, alpha=0.4); ax_prs.set_ylim(bottom=0)
    fig_m.tight_layout(pad=3); canvas_m.draw()

ttk.Button(ml, text="Motor simulieren ▶", command=run_motor).pack(padx=6, pady=8, fill=tk.X)

# ============================================================
# TAB 2: FLUGRECHNER (einstufig)
# ============================================================
tab_flight = ttk.Frame(nb)
nb.add(tab_flight, text="  Flugrechner  ")

fl = section(tab_flight, "Raketenparameter")
fl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8, ipadx=4, ipady=4)

flight_f = [
    ("Strukturmasse (g):",       "500"),
    ("Raketen-Durchmesser (mm):","65"),
    ("Luftwiderstand C_d:",      "0.35"),
    ("Starthöhe (m):",           "0"),
]
flight_entries = {}
for lbl_t, default in flight_f:
    lbl(fl, lbl_t).pack(anchor=tk.W, padx=6, pady=(5,0))
    e = entry(fl, default)
    e.pack(padx=6, fill=tk.X)
    flight_entries[lbl_t] = e

ttk.Separator(fl).pack(fill=tk.X, padx=6, pady=8)
flight_res = lbl(fl, "← Erst Motor simulieren", justify=tk.LEFT, foreground="#666")
flight_res.pack(padx=6)

fr = ttk.Frame(tab_flight)
fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=8)
fig_f, (ax_h, ax_v) = plt.subplots(2,1, figsize=(8,6))
fig_f.tight_layout(pad=3)
canvas_f = FigureCanvasTkAgg(fig_f, master=fr)
canvas_f.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def run_flight():
    if not thrust_data:
        flight_res.config(text="⚠ Erst Motor simulieren!"); return
    vals    = [e.get() for e in flight_entries.values()]
    ms      = float(vals[0])/1000
    diam    = float(vals[1])/1000
    Cd      = float(vals[2])
    h0      = float(vals[3])
    td      = thrust_data
    times,heights,vels = simulate_flight(
        td["times"],td["thrusts"],td["times"],td["prop_masses"],
        td["mp0"], ms, diam, h0, Cd)
    TWR = td["thrusts"].max() / ((ms+td["mp0"])*G0)
    hmax = max(heights); vmax = max(vels)
    flight_res.config(text=(
        f"Max Höhe:    {hmax:.0f} m ({hmax/1000:.2f} km)\n"
        f"Max Speed:   {vmax:.0f} m/s ({vmax/343:.2f} Mach)\n"
        f"TWR start:   {TWR:.2f}\n"
        f"Flugzeit:    {times[-1]:.1f} s\n"
        f"{'✓ >10km!' if hmax>=10000 else '✗ Ziel verfehlt'}"))
    ax_h.clear()
    ax_h.plot(times, np.array(heights)/1000, color="#457b9d", lw=2)
    for ref,col,lbl_t in [(10,"red","10 km"),(100,"purple","Kármán")]:
        ax_h.axhline(ref, color=col, ls="--", alpha=0.5, label=lbl_t)
    ax_h.set(xlabel="Zeit (s)", ylabel="Höhe (km)", title="Flugbahn")
    ax_h.legend(fontsize=9); ax_h.grid(True, alpha=0.4); ax_h.set_ylim(bottom=0)
    ax_v.clear()
    ax_v.plot(times, vels, color="#e76f51", lw=2)
    ax_v.axhline(343, color="gray", ls="--", alpha=0.5, label="Schall")
    ax_v.set(xlabel="Zeit (s)", ylabel="Geschwindigkeit (m/s)", title="Geschwindigkeit")
    ax_v.legend(fontsize=9); ax_v.grid(True, alpha=0.4)
    fig_f.tight_layout(pad=3); canvas_f.draw()

ttk.Button(fl, text="Flug simulieren ▶", command=run_flight).pack(padx=6, pady=8, fill=tk.X)

# ============================================================
# TAB 3: ZWEISTUFIG
# ============================================================
tab_2s = ttk.Frame(nb)
nb.add(tab_2s, text="  Zweistufig  ")

s2_left = ttk.Frame(tab_2s)
s2_left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=8)

# --- Stufe 1 ---
s1_frame = section(s2_left, "Stufe 1 – Motor")
s1_frame.pack(fill=tk.X, padx=4, pady=4, ipadx=4)

s1_motor_f = [
    ("Do (mm):","80"),("Di (mm):","32"),
    ("Lg (mm):","200"),("Segmente:","3"),("Dt (mm):","18"),
]
s1_entries = {}
for lbl_t, default in s1_motor_f:
    lbl(s1_frame, lbl_t).pack(anchor=tk.W, padx=4, pady=(3,0))
    e = entry(s1_frame, default, width=12)
    e.pack(padx=4, fill=tk.X)
    s1_entries[lbl_t] = e

lbl(s1_frame, "Strukturmasse S1 (g):").pack(anchor=tk.W, padx=4, pady=(3,0))
e_ms1 = entry(s1_frame, "600", width=12); e_ms1.pack(padx=4, fill=tk.X)
lbl(s1_frame, "Rumpfdurchmesser S1 (mm):").pack(anchor=tk.W, padx=4, pady=(3,0))
e_d1  = entry(s1_frame, "90",  width=12); e_d1.pack(padx=4, fill=tk.X)

# --- Stufe 2 ---
s2_frame = section(s2_left, "Stufe 2 – Motor")
s2_frame.pack(fill=tk.X, padx=4, pady=4, ipadx=4)

s2_motor_f = [
    ("Do (mm):","60"),("Di (mm):","24"),
    ("Lg (mm):","150"),("Segmente:","3"),("Dt (mm):","13"),
]
s2_entries = {}
for lbl_t, default in s2_motor_f:
    lbl(s2_frame, lbl_t).pack(anchor=tk.W, padx=4, pady=(3,0))
    e = entry(s2_frame, default, width=12)
    e.pack(padx=4, fill=tk.X)
    s2_entries[lbl_t] = e

lbl(s2_frame, "Strukturmasse S2 (g):").pack(anchor=tk.W, padx=4, pady=(3,0))
e_ms2 = entry(s2_frame, "300", width=12); e_ms2.pack(padx=4, fill=tk.X)
lbl(s2_frame, "Rumpfdurchmesser S2 (mm):").pack(anchor=tk.W, padx=4, pady=(3,0))
e_d2  = entry(s2_frame, "65",  width=12); e_d2.pack(padx=4, fill=tk.X)

# --- Gemeinsame Parameter ---
shared_frame = section(s2_left, "Allgemein")
shared_frame.pack(fill=tk.X, padx=4, pady=4, ipadx=4)

lbl(shared_frame, "C_d (beide Stufen):").pack(anchor=tk.W, padx=4, pady=(3,0))
e_cd2s = entry(shared_frame, "0.30", width=12); e_cd2s.pack(padx=4, fill=tk.X)
lbl(shared_frame, "Küstphase S1→S2 (s):").pack(anchor=tk.W, padx=4, pady=(3,0))
e_coast = entry(shared_frame, "0.5", width=12); e_coast.pack(padx=4, fill=tk.X)

ttk.Separator(s2_left).pack(fill=tk.X, padx=6, pady=6)
s2_res = lbl(s2_left, "", justify=tk.LEFT)
s2_res.pack(padx=6)

s2_right = ttk.Frame(tab_2s)
s2_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=8)
fig_2s, (ax_2h, ax_2v) = plt.subplots(2,1, figsize=(9,7))
fig_2s.tight_layout(pad=3)
canvas_2s = FigureCanvasTkAgg(fig_2s, master=s2_right)
canvas_2s.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def run_2stage():
    props = PROPELLANTS["KNSB 65/35"]
    # Stufe 1
    v1 = [e.get() for e in s1_entries.values()]
    Do1,Di1,Lg1 = float(v1[0])/1e3, float(v1[1])/1e3, float(v1[2])/1e3
    N1,Dt1 = int(float(v1[3])), float(v1[4])/1e3
    ms1   = float(e_ms1.get())/1000
    diam1 = float(e_d1.get())/1000
    t1,F1,pm1,mp01 = simulate_motor(Do1,Di1,Lg1,N1,Dt1,props)
    td1 = {"times":t1,"thrusts":F1,"prop_masses":pm1,"mp0":mp01}
    # Stufe 2
    v2 = [e.get() for e in s2_entries.values()]
    Do2,Di2,Lg2 = float(v2[0])/1e3, float(v2[1])/1e3, float(v2[2])/1e3
    N2,Dt2 = int(float(v2[3])), float(v2[4])/1e3
    ms2   = float(e_ms2.get())/1000
    diam2 = float(e_d2.get())/1000
    t2,F2,pm2,mp02 = simulate_motor(Do2,Di2,Lg2,N2,Dt2,props)
    td2 = {"times":t2,"thrusts":F2,"prop_masses":pm2,"mp0":mp02}
    Cd     = float(e_cd2s.get())
    coast  = float(e_coast.get())
    # TWR Check
    TWR1 = F1.max() / ((ms1+mp01+ms2+mp02)*G0)
    if TWR1 < 1.0:
        s2_res.config(text=f"⚠ TWR Stufe 1 = {TWR1:.2f} < 1\nMotor vergrößern!"); return
    times,heights,vels,h_sep,v_sep = simulate_2stage(
        td1,td2,ms1,diam1,ms2,diam2,coast,Cd)
    m_total = (ms1+mp01+ms2+mp02)
    hmax = max(heights); vmax = max(vels)
    imp1 = np.trapezoid(F1,t1); imp2 = np.trapezoid(F2,t2)
    s2_res.config(text=(
        f"Startmasse:   {m_total*1000:.0f} g\n"
        f"Treibstoff S1:{mp01*1000:.0f} g (Kl. {motor_class(imp1)})\n"
        f"Treibstoff S2:{mp02*1000:.0f} g (Kl. {motor_class(imp2)})\n"
        f"TWR (Start):  {TWR1:.2f}\n\n"
        f"Trennung bei: {h_sep/1000:.2f} km\n"
        f"Speed bei Tr.:{v_sep:.0f} m/s\n\n"
        f"Max Höhe:     {hmax/1000:.2f} km\n"
        f"Max Speed:    {vmax:.0f} m/s\n"
        f"             ({vmax/343:.2f} Mach)\n\n"
        f"{'✓✓ ZIEL >10km!' if hmax>=10000 else ('✓ Nah dran!' if hmax>=8000 else 'Motor anpassen')}"))
    sep_idx = np.searchsorted(times, t1[-1]+coast)
    ax_2h.clear()
    ax_2h.plot(times[:sep_idx], np.array(heights[:sep_idx])/1000,
               color="#e63946", lw=2, label="Stufe 1")
    ax_2h.plot(times[sep_idx:], np.array(heights[sep_idx:])/1000,
               color="#457b9d", lw=2, label="Stufe 2")
    ax_2h.axvline(times[sep_idx], color="gray", ls=":", alpha=0.7, label="Trennung")
    ax_2h.axhline(10, color="red", ls="--", alpha=0.5, label="10 km")
    ax_2h.axhline(100, color="purple", ls=":", alpha=0.4, label="Kármán")
    ax_2h.set(xlabel="Zeit (s)", ylabel="Höhe (km)", title="Zweistufige Flugbahn")
    ax_2h.legend(fontsize=9); ax_2h.grid(True, alpha=0.4); ax_2h.set_ylim(bottom=0)
    ax_2v.clear()
    ax_2v.plot(times[:sep_idx], vels[:sep_idx], color="#e63946", lw=2, label="Stufe 1")
    ax_2v.plot(times[sep_idx:], vels[sep_idx:], color="#457b9d", lw=2, label="Stufe 2")
    ax_2v.axvline(times[sep_idx], color="gray", ls=":", alpha=0.7)
    ax_2v.axhline(343, color="gray", ls="--", alpha=0.5, label="Schall")
    ax_2v.set(xlabel="Zeit (s)", ylabel="Geschwindigkeit (m/s)", title="Geschwindigkeit")
    ax_2v.legend(fontsize=9); ax_2v.grid(True, alpha=0.4)
    fig_2s.tight_layout(pad=3); canvas_2s.draw()

ttk.Button(s2_left, text="Zweistufig simulieren ▶",
           command=run_2stage).pack(padx=6, pady=8, fill=tk.X)

# ============================================================
# TAB 4: AERODYNAMIK & NOSECONE
# ============================================================
tab_aero = ttk.Frame(nb)
nb.add(tab_aero, text="  Aerodynamik  ")

aero_left = ttk.Frame(tab_aero)
aero_left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

nose_frame = section(aero_left, "Nosecone Design")
nose_frame.pack(fill=tk.X, padx=4, pady=4, ipadx=4)

lbl(nose_frame, "Form:").pack(anchor=tk.W, padx=4, pady=(5,0))
nose_var = tk.StringVar(value="Von Kármán Ogive")
ttk.OptionMenu(nose_frame, nose_var, "Von Kármán Ogive",
               "Kegelförmig","Ogive (klassisch)","Von Kármán Ogive",
               "Parabolisch","Elliptisch","Halbkugel").pack(padx=4, fill=tk.X)

aero_fields = [
    ("Fineness Ratio (L_nose/D):", "4.0"),
    ("Körper L/D Verhältnis:",     "15"),
    ("Körperdurchmesser (mm):",    "65"),
]
aero_entries = {}
for lbl_t, default in aero_fields:
    lbl(nose_frame, lbl_t).pack(anchor=tk.W, padx=4, pady=(5,0))
    e = entry(nose_frame, default)
    e.pack(padx=4, fill=tk.X)
    aero_entries[lbl_t] = e

ttk.Separator(aero_left).pack(fill=tk.X, padx=6, pady=6)
aero_res = lbl(aero_left, "", justify=tk.LEFT)
aero_res.pack(padx=6)

def calc_aero():
    shape    = nose_var.get()
    fineness = float(aero_entries["Fineness Ratio (L_nose/D):"].get())
    calibers = float(aero_entries["Körper L/D Verhältnis:"].get())
    diam     = float(aero_entries["Körperdurchmesser (mm):"].get())
    Cd = cd_nosecone(shape, fineness, calibers)
    l_nose = fineness * diam
    # Grafik
    ax_aero.clear()
    shapes = list({"Kegelförmig":0.55,"Ogive (klassisch)":0.38,
                   "Von Kármán Ogive":0.28,"Parabolisch":0.32,
                   "Elliptisch":0.42,"Halbkugel":0.65}.items())
    names, cds = zip(*shapes)
    colors = ["#e63946" if n==shape else "#adb5bd" for n in names]
    bars = ax_aero.bar(range(len(names)), cds, color=colors)
    ax_aero.set_xticks(range(len(names)))
    ax_aero.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax_aero.set_ylabel("Cd (geschätzt, subsonic)")
    ax_aero.set_title("Nosecone Cd Vergleich")
    ax_aero.axhline(Cd, color="#e63946", ls="--", alpha=0.5)
    ax_aero.grid(True, axis='y', alpha=0.4)
    ax_aero.set_ylim(0, 0.8)
    fig_aero.tight_layout(); canvas_aero.draw()
    aero_res.config(text=(
        f"Form:          {shape}\n"
        f"Fineness:      {fineness:.1f}\n"
        f"→ Cd:          {Cd:.3f}\n\n"
        f"Nosecone-Länge:{l_nose:.0f} mm\n\n"
        f"Empfehlung:\n"
        f"  Von Kármán Ogive\n"
        f"  Fineness 4-5\n"
        f"  → Cd ≈ 0.25-0.28"))

ttk.Button(aero_left, text="Cd berechnen ▶",
           command=calc_aero).pack(padx=6, pady=8, fill=tk.X)

aero_right = ttk.Frame(tab_aero)
aero_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=8)
fig_aero, ax_aero = plt.subplots(figsize=(7,5))
canvas_aero = FigureCanvasTkAgg(fig_aero, master=aero_right)
canvas_aero.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ============================================================
# TAB 5: STABILITÄT (Barrowman)
# ============================================================
tab_stab = ttk.Frame(nb)
nb.add(tab_stab, text="  Stabilität  ")

stab_left = section(tab_stab, "Barrowman Stabilitätsrechner")
stab_left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8, ipadx=4, ipady=4)

stab_fields = [
    ("Körperdurchmesser D (mm):",    "65"),
    ("Nasendurchmesser D_nose (mm):","65"),
    ("Nasenlänge L_nose (mm):",      "260"),
    ("Anzahl Fins:",                  "4"),
    ("Fin-Spanne s (mm):",           "80"),
    ("Fin-Wurzelsehne cr (mm):",     "100"),
    ("Fin-Spitzensehne ct (mm):",    "50"),
    ("Fin-Position ab Nase xt (mm):","900"),
    ("Gesamtlänge Rakete (mm):",     "1100"),
    ("Schwerpunkt CG ab Nase (mm):", "450"),
]
stab_entries = {}
for lbl_t, default in stab_fields:
    lbl(stab_left, lbl_t).pack(anchor=tk.W, padx=6, pady=(4,0))
    e = entry(stab_left, default)
    e.pack(padx=6, fill=tk.X)
    stab_entries[lbl_t] = e

ttk.Separator(stab_left).pack(fill=tk.X, padx=6, pady=8)
stab_res = lbl(stab_left, "", justify=tk.LEFT)
stab_res.pack(padx=6)

stab_right = ttk.Frame(tab_stab)
stab_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=8)
fig_stab, ax_stab = plt.subplots(figsize=(8,4))
canvas_stab = FigureCanvasTkAgg(fig_stab, master=stab_right)
canvas_stab.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def calc_stability():
    v = [e.get() for e in stab_entries.values()]
    D,Dn,Ln  = float(v[0])/1000, float(v[1])/1000, float(v[2])/1000
    Nf       = int(float(v[3]))
    s,cr,ct  = float(v[4])/1000, float(v[5])/1000, float(v[6])/1000
    xt,L_tot = float(v[7])/1000, float(v[8])/1000
    CG       = float(v[9])/1000
    Xcp, CN = barrowman_stability(D,Dn,Ln,Nf,s,cr,ct,xt,L_tot)
    stability = (Xcp - CG) / D
    verdict = ("✓ STABIL" if stability >= 1.5 else
               ("⚠ GRENZWERTIG" if stability >= 1.0 else "✗ INSTABIL"))
    stab_res.config(text=(
        f"Druckpunkt CP:  {Xcp*1000:.0f} mm ab Nase\n"
        f"Schwerpunkt CG: {CG*1000:.0f} mm ab Nase\n"
        f"CP - CG:        {(Xcp-CG)*1000:.0f} mm\n"
        f"Stabilität:     {stability:.2f} Kaliber\n\n"
        f"Bewertung: {verdict}\n\n"
        f"Empfehlung: 1.5-2.5 Kaliber\n"
        f"< 1.0 → Fins vergrößern\n"
        f"> 3.0 → Rakete überstabilisiert"))
    # Skizze
    ax_stab.clear()
    ax_stab.set_xlim(0, L_tot*1000+50)
    ax_stab.set_ylim(-D*600, D*600)
    ax_stab.set_aspect('equal')
    r = D/2*1000
    # Rumpf
    from matplotlib.patches import FancyArrowPatch, Rectangle
    rect = plt.Rectangle((Ln*1000,-(r)), (L_tot-Ln)*1000, 2*r,
                          color='#adb5bd', zorder=2)
    ax_stab.add_patch(rect)
    # Nosecone (Dreieck)
    ax_stab.fill([0, Ln*1000, Ln*1000],[0, r, -r], color='#457b9d', zorder=3)
    # Fins
    fin_x = xt*1000
    for sign in [1,-1]:
        ax_stab.fill([fin_x, fin_x+ct*1000, fin_x+cr*1000, fin_x],
                     [sign*r, sign*(r+s*1000), sign*(r+s*1000//2), sign*r],
                     color='#e63946', alpha=0.7, zorder=3)
    # CP & CG Markierungen
    ax_stab.axvline(Xcp*1000, color='blue', lw=2, label=f'CP={Xcp*1000:.0f}mm')
    ax_stab.axvline(CG*1000,  color='green',lw=2, label=f'CG={CG*1000:.0f}mm')
    col = "green" if stability>=1.5 else ("orange" if stability>=1 else "red")
    ax_stab.set_title(f"Stabilität: {stability:.2f} Kaliber  [{verdict}]", color=col)
    ax_stab.legend(fontsize=9); ax_stab.grid(True, alpha=0.3)
    ax_stab.set_xlabel("Position ab Nase (mm)")
    fig_stab.tight_layout(); canvas_stab.draw()

ttk.Button(stab_left, text="Stabilität berechnen ▶",
           command=calc_stability).pack(padx=6, pady=8, fill=tk.X)

# ============================================================
# TAB 6: TREIBSTOFFRECHNER
# ============================================================
tab_fuel = ttk.Frame(nb)
nb.add(tab_fuel, text="  Treibstoff  ")

fuel_left = section(tab_fuel, "Mischungsrechner")
fuel_left.pack(side=tk.LEFT, padx=8, pady=8, fill=tk.Y, ipadx=4, ipady=4)

fuel_fields = [
    ("Ziel-Treibstoffmasse (g):", "500"),
    ("KNO₃ Anteil (%):",          "65"),
    ("Sorbitol Anteil (%):",       "35"),
    ("Dichte (g/cm³):",            "1.75"),
]
fuel_entries = {}
for lbl_t, default in fuel_fields:
    lbl(fuel_left, lbl_t).pack(anchor=tk.W, padx=6, pady=(5,0))
    e = entry(fuel_left, default)
    e.pack(padx=6, fill=tk.X)
    fuel_entries[lbl_t] = e

fuel_res = lbl(fuel_left, "", justify=tk.LEFT)
fuel_res.pack(padx=6, pady=8)

def calc_fuel():
    v = [e.get() for e in fuel_entries.values()]
    total, ox, fu, dens = float(v[0]), float(v[1])/100, float(v[2])/100, float(v[3])
    fuel_res.config(text=(
        f"KNO₃:       {total*ox:.2f} g\n"
        f"Sorbitol:   {total*fu:.2f} g\n"
        f"Volumen:    {total/dens:.2f} cm³\n"
        f"           = {total/dens*1000:.1f} mm³"))

ttk.Button(fuel_left, text="Berechnen ▶",
           command=calc_fuel).pack(padx=6, pady=8, fill=tk.X)

fuel_right = section(tab_fuel, "KNSB Referenz")
fuel_right.pack(side=tk.LEFT, padx=8, pady=8, fill=tk.Y, ipadx=8, ipady=8)
lbl(fuel_right, (
    "KNSB 65/35\n──────────────────\n"
    "Isp:          ~150 s\n"
    "Brennrate:    ~7 mm/s @ 7 MPa\n"
    "Dichte:       ~1.75 g/cm³\n"
    "Verbr.temp.:  ~1600°C\n"
    "Gießtemp.:    160-170°C\n\n"
    "KNSB+Al 65/30/5\n──────────────────\n"
    "Isp:          ~160 s\n"
    "Verbr.temp.:  ~1800°C\n"
    "Cave: stat. Entladung!\n\n"
    "BATES Grain Faustregel:\n"
    "  Di ≈ 0.4 × Do\n"
    "  Dt ≈ 0.22 × Do\n"
    "  Min. Druck: ~0.5 MPa"
), font=("Courier",11), justify=tk.LEFT).pack(padx=4, pady=4)

# ============================================================
# TAB 7: RAKETENGEOMETRIE & MASSZEICHNUNG
# ============================================================
tab_geo = ttk.Frame(nb)
nb.add(tab_geo, text="  Geometrie  ")

geo_left = ttk.Frame(tab_geo)
geo_left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=8)

# --- Stufe 2 ---
gf_s2 = section(geo_left, "Stufe 2 – Geometrie")
gf_s2.pack(fill=tk.X, padx=4, pady=4, ipadx=4)

geo_s2_fields = [
    ("Rumpfdurchmesser (mm):",     "65"),
    ("Motorsegment-Länge (mm):",   "490"),
    ("Anzahl Segmente:",           "3"),
    ("Nutzlastsektion (mm):",      "150"),
    ("Nosecone Fineness (L/D):",   "4.5"),
    ("Fin-Wurzelsehne cr (mm):",   "80"),
    ("Fin-Spanne s (mm):",         "60"),
    ("Fin-Dicke (mm):",            "4"),
    ("Masse Stufe 2 (g):",         "2245"),
]
geo_s2_entries = {}
for lbl_t, default in geo_s2_fields:
    lbl(gf_s2, lbl_t).pack(anchor=tk.W, padx=4, pady=(3,0))
    e = entry(gf_s2, default, width=12)
    e.pack(padx=4, fill=tk.X)
    geo_s2_entries[lbl_t] = e

# --- Stufe 1 ---
gf_s1 = section(geo_left, "Stufe 1 – Geometrie")
gf_s1.pack(fill=tk.X, padx=4, pady=4, ipadx=4)

geo_s1_fields = [
    ("Rumpfdurchmesser (mm):",     "85"),
    ("Motorsegment-Länge (mm):",   "640"),
    ("Anzahl Segmente:",           "3"),
    ("Adapter-Länge (mm):",        "100"),
    ("Fin-Wurzelsehne cr (mm):",   "120"),
    ("Fin-Spanne s (mm):",         "90"),
    ("Fin-Dicke (mm):",            "5"),
    ("Masse Stufe 1 (g):",         "4965"),
]
geo_s1_entries = {}
for lbl_t, default in geo_s1_fields:
    lbl(gf_s1, lbl_t).pack(anchor=tk.W, padx=4, pady=(3,0))
    e = entry(gf_s1, default, width=12)
    e.pack(padx=4, fill=tk.X)
    geo_s1_entries[lbl_t] = e

ttk.Separator(geo_left).pack(fill=tk.X, padx=6, pady=6)
geo_res = lbl(geo_left, "", justify=tk.LEFT, font=("Courier", 10))
geo_res.pack(padx=6)

# --- Canvas für Zeichnung ---
geo_right = ttk.Frame(tab_geo)
geo_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=8)

fig_geo, ax_geo = plt.subplots(figsize=(10, 7))
canvas_geo = FigureCanvasTkAgg(fig_geo, master=geo_right)
canvas_geo.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def draw_geometry():
    from matplotlib.patches import Polygon, FancyArrowPatch
    import matplotlib.gridspec as gridspec

    # --- Eingaben lesen ---
    s2 = {k: float(e.get()) for k, e in geo_s2_entries.items()}
    s1 = {k: float(e.get()) for k, e in geo_s1_entries.items()}

    D2   = s2["Rumpfdurchmesser (mm):"]
    Lm2  = s2["Motorsegment-Länge (mm):"]
    N2   = int(s2["Anzahl Segmente:"])
    Lp2  = s2["Nutzlastsektion (mm):"]
    fn2  = s2["Nosecone Fineness (L/D):"]
    cr2  = s2["Fin-Wurzelsehne cr (mm):"]
    s_2  = s2["Fin-Spanne s (mm):"]
    m2   = s2["Masse Stufe 2 (g):"]

    D1   = s1["Rumpfdurchmesser (mm):"]
    Lm1  = s1["Motorsegment-Länge (mm):"]
    N1   = int(s1["Anzahl Segmente:"])
    La1  = s1["Adapter-Länge (mm):"]
    cr1  = s1["Fin-Wurzelsehne cr (mm):"]
    s_1  = s1["Fin-Spanne s (mm):"]
    m1   = s1["Masse Stufe 1 (g):"]

    # Berechnete Längen
    Ln2      = fn2 * D2
    Lmotor2  = Lm2 * N2
    L2_total = Ln2 + Lp2 + Lmotor2
    Lmotor1  = Lm1 * N1
    L1_total = Lmotor1 + La1
    L_gesamt = L1_total + L2_total
    m_gesamt = m1 + m2

    # Abschnitts-Positionen
    x_nose   = 0
    x_pay    = Ln2
    x_mot2   = x_pay   + Lp2
    x_sep    = x_mot2  + Lmotor2
    x_mot1   = x_sep   + La1
    x_end    = x_mot1  + Lmotor1

    # --- Zeichnung ---
    ax_geo.clear()
    BG = '#16213e'
    ax_geo.set_facecolor(BG)
    fig_geo.patch.set_facecolor(BG)

    r2 = D2 / 2
    r1 = D1 / 2

    C_NOSE    = '#2ec4b6'
    C_PAYLOAD = '#f4a261'
    C_MOT2    = '#e63946'
    C_MOT2L   = '#ff6b6b'
    C_ADAPTER = '#6c757d'
    C_MOT1    = '#9b5de5'
    C_MOT1L   = '#c77dff'
    C_FIN     = '#a8dadc'
    C_NOZZLE  = '#495057'
    C_SEP     = '#ffd166'
    C_DIM     = '#8ecae6'
    C_WHITE   = '#f8f9fa'

    # Dynamische Y-Achse: genug Platz für Fins + Bemaßung oben/unten
    fin_max   = max(r1 + s_1, r2 + s_2)
    dim_top   = fin_max + 90   # Platz für 3 Bemaßungszeilen
    dim_bot   = -(r1 + s_1 + 50)
    margin_x  = L_gesamt * 0.04

    ax_geo.set_xlim(-margin_x, L_gesamt + margin_x)
    ax_geo.set_ylim(dim_bot, dim_top)
    ax_geo.set_aspect('equal')
    ax_geo.axis('off')

    # ── Hilfsfunktionen ──────────────────────────────────────
    def trapez(x, w, r_l, r_r, color, alpha=0.88, lw=0.7):
        poly = Polygon([[x,r_l],[x+w,r_r],[x+w,-r_r],[x,-r_l]],
                       closed=True, facecolor=color, edgecolor='white',
                       alpha=alpha, linewidth=lw, zorder=3)
        ax_geo.add_patch(poly)

    def seg_label(x, w, r, text, color='white', fs=7):
        """Label NUR wenn genug Platz (Mindestbreite 40mm)."""
        if w >= 40:
            ax_geo.text(x + w/2, 0, text, color=color, fontsize=fs,
                        ha='center', va='center', fontweight='bold',
                        zorder=5, clip_on=True)

    def dim_line(x1, x2, y, text, color=C_DIM, fs=7.5, tick_len=6):
        """Bemaßungslinie mit Ticks und Text darüber."""
        ax_geo.annotate('', xy=(x2,y), xytext=(x1,y),
                        arrowprops=dict(arrowstyle='<->', color=color,
                                        lw=1.2, mutation_scale=8), zorder=6)
        # Vertikale Ticks
        ax_geo.plot([x1,x1],[y-tick_len/2, y+tick_len/2], color=color, lw=0.8, zorder=6)
        ax_geo.plot([x2,x2],[y-tick_len/2, y+tick_len/2], color=color, lw=0.8, zorder=6)
        ax_geo.text((x1+x2)/2, y + tick_len, text, color=color, fontsize=fs,
                    ha='center', va='bottom', zorder=7,
                    bbox=dict(facecolor=BG, alpha=0.7, pad=1, edgecolor='none'))

    def connect_dim(x, y_from, y_to, color=C_DIM):
        ax_geo.plot([x,x],[y_from,y_to], color=color, lw=0.6, ls=':', zorder=2)

    # ── NOSECONE ────────────────────────────────────────────
    poly_nose = Polygon([[x_nose,0],[x_pay,r2],[x_pay,-r2]],
                        closed=True, facecolor=C_NOSE, edgecolor='white',
                        alpha=0.9, lw=0.8, zorder=3)
    ax_geo.add_patch(poly_nose)
    if Ln2 >= 80:
        ax_geo.text(x_nose + Ln2*0.55, 0, "Nosecone", color='white',
                    fontsize=7, ha='center', va='center', fontweight='bold', zorder=5)

    # ── NUTZLAST ────────────────────────────────────────────
    trapez(x_pay, Lp2, r2, r2, C_PAYLOAD)
    seg_label(x_pay, Lp2, r2, "Nutzlast", fs=6.5)

    # ── MOTOR STUFE 2 ────────────────────────────────────────
    seg_cols2 = [C_MOT2, C_MOT2L, '#ff8fa3', '#ffb3c1']
    for i in range(N2):
        xi = x_mot2 + i * Lm2
        trapez(xi, Lm2, r2, r2, seg_cols2[i % len(seg_cols2)])
        seg_label(xi, Lm2, r2, f"S2·{i+1}", fs=6)
        if i < N2-1:
            ax_geo.plot([xi+Lm2]*2, [-r2, r2], color='white', lw=1.2, zorder=4)

    # ── FINS STUFE 2 ────────────────────────────────────────
    xfr2 = x_mot2 + Lmotor2 - cr2
    for sgn in [1,-1]:
        fin = [[xfr2, sgn*r2],
               [xfr2 + cr2*0.55, sgn*(r2+s_2)],
               [xfr2 + cr2,      sgn*(r2+s_2*0.35)],
               [xfr2 + cr2,      sgn*r2]]
        ax_geo.add_patch(Polygon(fin, closed=True, facecolor=C_FIN,
                                 edgecolor='white', alpha=0.85, lw=0.7, zorder=2))

    # ── ADAPTER ─────────────────────────────────────────────
    trapez(x_sep, La1, r2, r1, C_ADAPTER)
    if La1 >= 60:
        ax_geo.text(x_sep + La1/2, 0, "Adapter", color='white',
                    fontsize=6, ha='center', va='center', zorder=5)

    # Trennlinie
    ax_geo.plot([x_sep]*2, [-r2-s_2*0.3, r2+s_2*0.3],
                color=C_SEP, lw=2.5, ls='--', zorder=5)

    # ── MOTOR STUFE 1 ────────────────────────────────────────
    seg_cols1 = [C_MOT1, C_MOT1L, '#d4a8ff', '#e9c7ff']
    for i in range(N1):
        xi = x_mot1 + i * Lm1
        trapez(xi, Lm1, r1, r1, seg_cols1[i % len(seg_cols1)])
        seg_label(xi, Lm1, r1, f"S1·{i+1}", fs=6)
        if i < N1-1:
            ax_geo.plot([xi+Lm1]*2, [-r1, r1], color='white', lw=1.2, zorder=4)

    # ── FINS STUFE 1 ────────────────────────────────────────
    xfr1 = x_mot1 + Lmotor1 - cr1
    for sgn in [1,-1]:
        fin = [[xfr1, sgn*r1],
               [xfr1 + cr1*0.5, sgn*(r1+s_1)],
               [xfr1 + cr1,     sgn*(r1+s_1*0.4)],
               [xfr1 + cr1,     sgn*r1]]
        ax_geo.add_patch(Polygon(fin, closed=True, facecolor=C_FIN,
                                 edgecolor='white', alpha=0.9, lw=0.7, zorder=2))

    # ── DÜSE ────────────────────────────────────────────────
    noz_l = min(45, r1*0.8)
    noz = [[x_end, r1],[x_end+noz_l, r1*0.35],
           [x_end+noz_l,-r1*0.35],[x_end,-r1]]
    ax_geo.add_patch(Polygon(noz, closed=True, facecolor=C_NOZZLE,
                             edgecolor='white', alpha=0.9, lw=0.8, zorder=3))

    # ── BESCHRIFTUNGEN (außerhalb, mit Pfeilen) ──────────────
    # Nosecone Label
    ax_geo.annotate("Nosecone\nVon Kármán\n"f"{Ln2:.0f}mm",
                    xy=(x_nose + Ln2*0.4, r2*0.5),
                    xytext=(x_nose + Ln2*0.2, r2 + s_2 + 28),
                    color=C_NOSE, fontsize=7, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='->', color=C_NOSE, lw=0.8),
                    zorder=8,
                    bbox=dict(facecolor=BG, alpha=0.75, pad=2, edgecolor='none'))

    # Nutzlast Label
    ax_geo.annotate("Nutzlast\nESP32+Baro\nFallschirm",
                    xy=(x_pay + Lp2/2, r2),
                    xytext=(x_pay + Lp2/2, r2 + s_2 + 28),
                    color=C_PAYLOAD, fontsize=7, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='->', color=C_PAYLOAD, lw=0.8),
                    zorder=8,
                    bbox=dict(facecolor=BG, alpha=0.75, pad=2, edgecolor='none'))

    # Trennung Label
    ax_geo.annotate("TRENNUNG",
                    xy=(x_sep, r2 + 5),
                    xytext=(x_sep, r1 + s_1 + 28),
                    color=C_SEP, fontsize=7.5, ha='center', va='bottom',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_SEP, lw=1.0),
                    zorder=8,
                    bbox=dict(facecolor=BG, alpha=0.8, pad=2,
                              edgecolor=C_SEP, linewidth=0.8))

    # ── BEMAССUNGSLINIEN (3 Ebenen) ──────────────────────────
    y_d1 = fin_max + 18   # Nosecone / einzelne Sektionen
    y_d2 = fin_max + 42   # Stufe 2 / Stufe 1
    y_d3 = fin_max + 68   # Gesamt

    # Nosecone
    connect_dim(x_nose,  r2, y_d1-4, C_NOSE)
    connect_dim(x_pay,   r2, y_d1-4, C_NOSE)
    dim_line(x_nose, x_pay, y_d1, f"NC {Ln2:.0f}", C_NOSE, fs=6.5, tick_len=4)

    # Nutzlast
    connect_dim(x_pay,   r2, y_d1-4, C_PAYLOAD)
    connect_dim(x_mot2,  r2, y_d1-4, C_PAYLOAD)
    dim_line(x_pay, x_mot2, y_d1, f"NL {Lp2:.0f}", C_PAYLOAD, fs=6.5, tick_len=4)

    # Stufe 2 gesamt
    connect_dim(x_nose, r2, y_d2-4, C_NOSE)
    connect_dim(x_sep,  r1, y_d2-4, C_NOSE)
    dim_line(x_nose, x_sep, y_d2, f"Stufe 2: {L2_total:.0f} mm", C_NOSE, fs=7.5)

    # Stufe 1 gesamt
    connect_dim(x_sep, r1, y_d2-4, C_MOT1L)
    connect_dim(x_end, r1, y_d2-4, C_MOT1L)
    dim_line(x_sep, x_end, y_d2, f"Stufe 1: {L1_total:.0f} mm", C_MOT1L, fs=7.5)

    # Gesamt
    connect_dim(x_nose, r2, y_d3-4, C_WHITE)
    connect_dim(x_end,  r1, y_d3-4, C_WHITE)
    dim_line(x_nose, x_end, y_d3,
             f"GESAMT: {L_gesamt:.0f} mm  =  {L_gesamt/1000:.2f} m",
             C_WHITE, fs=8.5)

    # ── DURCHMESSER-PFEILE (unten) ───────────────────────────
    xd2 = x_mot2 + Lm2/2
    ax_geo.annotate('', xy=(xd2, r2), xytext=(xd2, -r2),
                    arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=1.0))
    ax_geo.text(xd2 - r2*0.6, 0, f"⌀{D2:.0f}", color=C_DIM,
                fontsize=7, ha='right', va='center', zorder=5,
                bbox=dict(facecolor=BG, alpha=0.7, pad=1, edgecolor='none'))

    xd1 = x_mot1 + Lm1/2
    ax_geo.annotate('', xy=(xd1, r1), xytext=(xd1, -r1),
                    arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=1.0))
    ax_geo.text(xd1 - r1*0.5, 0, f"⌀{D1:.0f}", color=C_DIM,
                fontsize=7, ha='right', va='center', zorder=5,
                bbox=dict(facecolor=BG, alpha=0.7, pad=1, edgecolor='none'))

    # ── LEGENDE (unten links) ────────────────────────────────
    leg_items = [
        (C_NOSE,    f"Nosecone  {Ln2:.0f} mm"),
        (C_PAYLOAD, f"Nutzlast  {Lp2:.0f} mm"),
        (C_MOT2,    f"Motor S2  {Lmotor2:.0f} mm  ({mp2:.0f} g)"),
        (C_MOT1,    f"Motor S1  {Lmotor1:.0f} mm  ({mp1:.0f} g)"),
        (C_ADAPTER, f"Adapter   {La1:.0f} mm"),
        (C_FIN,     f"Fins S2 span {s_2:.0f}  |  S1 span {s_1:.0f} mm"),
    ]
    # Berechne Treibstoffmassen für Legende
    rho = 1750
    mp2 = rho * N2 * np.pi/4 * (geo_s2_entries["Rumpfdurchmesser (mm):"].get() == geo_s2_entries["Rumpfdurchmesser (mm):"].get() and 1) * 0  # placeholder, use m2 directly
    # Simpel: zeige einfach nur Treibstoffanteil
    tp = (m2+m1)*0
    leg_y_start = -(r1 + s_1 + 12)
    for i,(col,txt) in enumerate(leg_items):
        ax_geo.text(x_nose, leg_y_start - i*11, f"■ {txt}",
                    color=col, fontsize=6.8, va='top', zorder=8,
                    bbox=dict(facecolor=BG, alpha=0.0, pad=0, edgecolor='none'))

    # Masse-Zusammenfassung rechts unten
    ax_geo.text(x_end, leg_y_start,
                f"Gesamtmasse: {m_gesamt/1000:.2f} kg\n"
                f"Stufe 2: {m2/1000:.2f} kg  ⌀{D2:.0f}mm\n"
                f"Stufe 1: {m1/1000:.2f} kg  ⌀{D1:.0f}mm\n"
                f"L/D gesamt: {L_gesamt/D1:.1f}",
                color=C_WHITE, fontsize=7.5, va='top', ha='right', zorder=8,
                bbox=dict(facecolor='#0f3460', alpha=0.85, pad=5,
                          edgecolor='#4a90d9', linewidth=0.8))

    ax_geo.set_title("Raketengeometrie  –  Maßzeichnung (Seitenansicht)",
                     color=C_WHITE, fontsize=11, pad=8, fontweight='bold')

    # Ergebnistext (linkes Panel)
    geo_res.config(text=(
        f"Gesamtlänge:  {L_gesamt:.0f} mm ({L_gesamt/1000:.2f} m)\n"
        f"Gesamtmasse:  {m_gesamt:.0f} g ({m_gesamt/1000:.2f} kg)\n\n"
        f"Stufe 2:      {L2_total:.0f} mm\n"
        f"  Nosecone:   {Ln2:.0f} mm\n"
        f"  Nutzlast:   {Lp2:.0f} mm\n"
        f"  Motor:      {Lmotor2:.0f} mm\n\n"
        f"Stufe 1:      {L1_total:.0f} mm\n"
        f"  Motor:      {Lmotor1:.0f} mm\n"
        f"  Adapter:    {La1:.0f} mm\n\n"
        f"L/D Gesamt:   {L_gesamt/D1:.1f}\n"
        f"L/D Stufe 2:  {L2_total/D2:.1f}"
    ))

    fig_geo.tight_layout(pad=0.5)
    canvas_geo.draw()

ttk.Button(geo_left, text="Zeichnung generieren ▶",
           command=draw_geometry).pack(padx=6, pady=10, fill=tk.X)

# Beim Start direkt zeichnen
root.after(200, draw_geometry)

# ============================================================
root.mainloop()
