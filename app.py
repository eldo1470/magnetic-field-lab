# ============================================================
# 🧲 INTERACTIVE MAGNETIC FIELD LAB
# 🔥 TRUE 3D EQUILIBRIUM SOLVER - PHYSICAL REALITY EDITION
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

from scipy.ndimage import minimum_filter
from scipy.optimize import root

# ============================================================
# STREAMLIT SETTINGS
# ============================================================

st.set_page_config(
    page_title="Magnetic Field Lab",
    layout="wide",
    page_icon="🧲"
)

st.title("🧲 TRUE 3D MAGNETIC EQUILIBRIUM LAB")

st.markdown("""
이 버전은 단순한 자기장 최소점 탐색이 아니라, 실제 물리학적 자기 쌍극자 모형과 
자석 내부 차단 마스킹 기법을 도입하여 현실에 존재하는 **진짜 3D 평형점**만 검출합니다.

### 평형 조건
Bx = 0, By = 0, Bz = 0 을 동시에 만족하는 공간상의 절대 영점
""")

# ============================================================
# PHYSICS ENGINE
# ============================================================

class MagneticFieldSolver:

    # --------------------------------------------------------
    # 단극자 모델 (바 자석의 양 끝단 근사용)
    # --------------------------------------------------------
    @staticmethod
    def calc_monopole(X, Y, Z, pole_pos, pole_strength):
        px, py, pz = pole_pos
        dx = X - px
        dy = Y - py
        dz = Z - pz

        # singularity stabilization
        r2 = dx**2 + dy**2 + dz**2
        r2 = np.maximum(r2, 1e-6)
        r = np.sqrt(r2)

        Bx = pole_strength * dx / (r**3)
        By = pole_strength * dy / (r**3)
        Bz = pole_strength * dz / (r**3)

        return Bx, By, Bz

    # --------------------------------------------------------
    # 점 및 배열 자석 내부 판정 함수 (가짜 내부분점 제거용)
    # --------------------------------------------------------
    @classmethod
    def is_inside_magnet(cls, x, y, z, magnets):
        for mag in magnets:
            x0, y0 = mag["pos"]
            angle_rad = np.radians(mag["angle"])
            mx = np.cos(angle_rad)
            my = np.sin(angle_rad)
            size = mag["size"]

            if mag["type"] == "Bar Magnet":
                # 시각화 사각형 중심 좌표 역산
                cx = x0 - mx * (size / 2)
                cy = y0 - my * (size / 2)
                dx = x - cx
                dy = y - cy
                
                # 자석 중심 기준 로컬 좌표계로 회전 변환
                x_local = dx * mx + dy * my
                y_local = -dx * my + dy * mx
                
                half_l = size / 2
                half_w = (size * 0.3) / 2
                half_h = (size * 0.3) / 2  # 3D 두께 가정
                
                if (abs(x_local) <= half_l and abs(y_local) <= half_w and abs(z) <= half_h):
                    return True
            else:
                # 구형 자석 중심 좌표 역산
                radius = size * 0.2
                cx = x0 - mx * radius
                cy = y0 - my * radius
                r = np.sqrt((x - cx)**2 + (y - cy)**2 + z**2)
                if r <= radius:
                    return True
        return False

    # --------------------------------------------------------
    # 전체 자기장 계산 (현실적 쌍극자 모델 반영)
    # --------------------------------------------------------
    @classmethod
    def get_field_from_magnets(cls, X, Y, Z, magnets):
        Bx_tot = np.zeros_like(X, dtype=float)
        By_tot = np.zeros_like(Y, dtype=float)
        Bz_tot = np.zeros_like(Z, dtype=float)

        for mag in magnets:
            x0, y0 = mag["pos"]
            angle_rad = np.radians(mag["angle"])
            mx = np.cos(angle_rad)
            my = np.sin(angle_rad)
            strength = mag["strength"]
            size = mag["size"]

            # =================================================
            # BAR MAGNET (정밀화된 유한 쌍극자 모델)
            # =================================================
            if mag["type"] == "Bar Magnet":
                Nx = x0
                Ny = y0
                Sx = x0 - mx * size
                Sy = y0 - my * size

                BxN, ByN, BzN = cls.calc_monopole(X, Y, Z, (Nx, Ny, 0), strength)
                BxS, ByS, BzS = cls.calc_monopole(X, Y, Z, (Sx, Sy, 0), -strength)
                
                Bx_tot += (BxN + BxS)
                By_tot += (ByN + ByS)
                Bz_tot += (BzN + BzS)

            # =================================================
            # SPHERE MAGNET (물리적으로 완벽한 이상적 자기 쌍극자)
            # =================================================
            else:
                radius = size * 0.2
                cx = x0 - mx * radius
                cy = y0 - my * radius

                eff_strength = strength * (size / 0.1)
                m_mag = eff_strength * 0.1  # 쌍극자 모멘트 크기 산정
                
                rx = X - cx
                ry = Y - cy
                rz = Z
                
                r2 = rx**2 + ry**2 + rz**2
                r2 = np.maximum(r2, 1e-6)
                r = np.sqrt(r2)
                r5 = r**5
                
                # m 벡터와 r 벡터의 내적
                mdotr = m_mag * (mx * rx + my * ry)
                
                # 이상적 자기 쌍극자 공식 적용
                Bx_tot += (3 * mdotr * rx - r2 * (m_mag * mx)) / r5
                By_tot += (3 * mdotr * ry - r2 * (m_mag * my)) / r5
                Bz_tot += (3 * mdotr * rz) / r5

        # NaN / Inf 안정화
        Bx_tot = np.nan_to_num(Bx_tot)
        By_tot = np.nan_to_num(By_tot)
        Bz_tot = np.nan_to_num(Bz_tot)

        return Bx_tot, By_tot, Bz_tot

    # --------------------------------------------------------
    # root solver 벡터 함수
    # --------------------------------------------------------
    @classmethod
    def field_vector_3d(cls, pos_3d, magnets):
        bx, by, bz = cls.get_field_from_magnets(
            pos_3d[0], pos_3d[1], pos_3d[2], magnets
        )
        return np.array([bx, by, bz])

    # --------------------------------------------------------
    # TRUE 3D EQUILIBRIUM FINDER (자석 내부 필터링 적용)
    # --------------------------------------------------------
    @classmethod
    def find_true_3d_equilibria(cls, magnets, initial_guesses):
        true_equilibria = []
        found_coords = []
        MAX_COORD = 50
        COMPONENT_TOL = 1e-5

        for guess in initial_guesses:
            try:
                result = root(
                    cls.field_vector_3d,
                    guess,
                    args=(magnets,),
                    method='hybr',
                    tol=1e-8
                )
            except Exception:
                continue

            if not result.success:
                continue

            ex, ey, ez = result.x

            if not np.all(np.isfinite(result.x)):
                continue

            if (abs(ex) > MAX_COORD or abs(ey) > MAX_COORD or abs(ez) > MAX_COORD):
                continue

            # 물리적 필터: 만약 해가 자석 내부 공간에 존재한다면 허상이므로 제외
            if cls.is_inside_magnet(ex, ey, ez, magnets):
                continue

            b_vector = cls.field_vector_3d(result.x, magnets)
            bx = float(b_vector[0])
            by = float(b_vector[1])
            bz = float(b_vector[2])

            # 진짜 컴포넌트 제로 검증
            if not (abs(bx) < COMPONENT_TOL and abs(by) < COMPONENT_TOL and abs(bz) < COMPONENT_TOL):
                continue

            # 중복 제거
            is_duplicate = False
            for fx, fy, fz in found_coords:
                dist = np.sqrt((ex-fx)**2 + (ey-fy)**2 + (ez-fz)**2)
                if dist < 0.05:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            found_coords.append((ex, ey, ez))
            true_equilibria.append({
                'pos': (ex, ey, ez),
                'Bx': bx,
                'By': by,
                'Bz': bz,
                'max_component': max(abs(bx), abs(by), abs(bz))
            })

        return true_equilibria

# ============================================================
# SIDEBAR (UI 유지 보존)
# ============================================================

with st.sidebar:
    st.header("⚙️ Simulation Settings")
    grid_size = st.slider("Grid Resolution", 100, 350, 180)
    field_range = st.slider("Field Range", 2.0, 15.0, 6.0)
    z_obs = st.slider("Observation Plane Z", -2.0, 2.0, 0.5, step=0.05)
    st.markdown("---")

    show_streamlines = st.checkbox("Streamlines", True)
    show_heatmap = st.checkbox("Bz Heatmap", True)
    show_bx_contour = st.checkbox("Bx = 0", True)
    show_by_contour = st.checkbox("By = 0", True)
    show_bz_contour = st.checkbox("Bz = 0", True)
    show_eq_2d = st.checkbox("2D Candidate Points", False)
    show_eq_3d = st.checkbox("True 3D Equilibria", True)

    st.header("🧲 Magnets")
    num_magnets = st.number_input("Number of Magnets", min_value=1, max_value=6, value=2)

    magnets = []
    default_configs = [
        {"pos": (-1.5, 0.0), "angle": 0},
        {"pos": (1.5, 0.0), "angle": 180},
    ]

    for i in range(int(num_magnets)):
        with st.expander(f"Magnet {i+1}", expanded=(i < 2)):
            def_pos = default_configs[i]["pos"] if i < len(default_configs) else (0.0, 0.0)
            def_ang = default_configs[i]["angle"] if i < len(default_configs) else 0

            m_type = st.selectbox("Type", ["Bar Magnet", "Sphere Magnet"], key=f"type_{i}")
            col1, col2 = st.columns(2)

            with col1:
                x0 = st.number_input("X", -field_range, field_range, def_pos[0], step=0.1, key=f"x_{i}")
                strength = st.slider("Strength", 0.1, 5.0, 1.0, step=0.1, key=f"str_{i}")
            with col2:
                y0 = st.number_input("Y", -field_range, field_range, def_pos[1], step=0.1, key=f"y_{i}")
                size = st.slider("Size", 0.2, 3.0, 1.0, step=0.1, key=f"size_{i}")

            angle = st.slider("Angle", 0, 360, def_ang, key=f"ang_{i}")

            magnets.append({
                "type": m_type,
                "pos": (x0, y0),
                "angle": angle,
                "strength": strength,
                "size": size
            })

# ============================================================
# GRID
# ============================================================

x_coords = np.linspace(-field_range, field_range, grid_size)
y_coords = np.linspace(-field_range, field_range, grid_size)
X, Y = np.meshgrid(x_coords, y_coords)
Z = np.full_like(X, z_obs)

# ============================================================
# FIELD COMPUTATION
# ============================================================

Bx_tot, By_tot, Bz_tot = MagneticFieldSolver.get_field_from_magnets(X, Y, Z, magnets)

B_mag = np.sqrt(Bx_tot**2 + By_tot**2 + Bz_tot**2)
B_xy_mag = np.sqrt(Bx_tot**2 + By_tot**2)

B_mag = np.nan_to_num(B_mag)
B_xy_mag = np.nan_to_num(B_xy_mag)

# ============================================================
# 2D CANDIDATES
# ============================================================

local_min_xy = (B_xy_mag == minimum_filter(B_xy_mag, size=5))
threshold = np.max(B_xy_mag) * 0.08
eq_2d_mask = local_min_xy & (B_xy_mag < threshold)

eq_2d_x_arr = X[eq_2d_mask]
eq_2d_y_arr = Y[eq_2d_mask]

# ============================================================
# ROOT FINDING (사각지대 방지 그리드 알고리즘 고도화)
# ============================================================

initial_guesses = []

# 2D 후보 평면 기반 추측값 배치
for ex, ey in zip(eq_2d_x_arr, eq_2d_y_arr):
    initial_guesses.append([ex, ey, z_obs])
    initial_guesses.append([ex, ey, 0.0])
    initial_guesses.append([ex, ey, -z_obs])

# 입체 사각지대(Blind Spots)를 예방하기 위한 전체 공간 격자 추측값 추가
coarse_grid = np.linspace(-field_range * 0.7, field_range * 0.7, 4)
for gx in coarse_grid:
    for gy in coarse_grid:
        for gz in [-1.2, -0.3, 0.3, 1.2]:
            initial_guesses.append([gx, gy, gz])

true_3d_equilibria = MagneticFieldSolver.find_true_3d_equilibria(magnets, initial_guesses)

# ============================================================
# VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='#0e1117')
ax.set_facecolor('#0e1117')

# HEATMAP
if show_heatmap:
    ax.imshow(
        np.abs(Bz_tot),
        extent=[-field_range, field_range, -field_range, field_range],
        origin='lower',
        cmap='magma',
        alpha=0.8
    )

# STREAMLINES
if show_streamlines:
    safe_B = np.nan_to_num(B_xy_mag, nan=0.0, posinf=0.0, neginf=0.0)
    max_b = np.max(safe_B) + 1e-12
    lw = 1.5 * (safe_B / max_b)**0.25

    ax.streamplot(
        x_coords, y_coords, Bx_tot, By_tot,
        color=(1, 1, 1, 0.4),
        linewidth=lw,
        density=1.5,
        arrowsize=1.0
    )

# NULLCLINES
try:
    if show_bx_contour:
        ax.contour(X, Y, Bx_tot, levels=[0], colors='#FF3333', linewidths=2.0, linestyles='--')
    if show_by_contour:
        ax.contour(X, Y, By_tot, levels=[0], colors='#3399FF', linewidths=2.0, linestyles='--')
    if show_bz_contour:
        ax.contour(X, Y, Bz_tot, levels=[0], colors='white', linewidths=2.5)
except Exception:
    pass

# 2D CANDIDATES DRAWING
if show_eq_2d:
    ax.scatter(
        eq_2d_x_arr, eq_2d_y_arr,
        color='#FF8800', edgecolors='white',
        s=80, zorder=5, label='2D Candidates'
    )

# TRUE 3D EQUILIBRIA DRAWING
if show_eq_3d:
    for eq in true_3d_equilibria:
        ex, ey, ez = eq['pos']
        bx, by, bz = eq['Bx'], eq['By'], eq['Bz']

        if (abs(ex) > field_range * 2 or abs(ey) > field_range * 2):
            continue

        ax.scatter(
            ex, ey,
            color='#FFD700', edgecolors='white',
            s=350, zorder=8, marker='*'
        )

        text_x = np.clip(ex + 0.25, -field_range, field_range)
        text_y = np.clip(ey + 0.25, -field_range, field_range)

        ax.text(
            text_x, text_y,
            (f"Z={ez:.3f}\n" f"Bx={bx:.1e}\n" f"By={by:.1e}\n" f"Bz={bz:.1e}"),
            color='#FFD700', fontsize=8, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none')
        )

# MAGNET DRAWING (UI 싱크 유지를 위해 기존 드로잉 로직 원형 유지)
for mag in magnets:
    x0, y0 = mag["pos"]
    angle_rad = np.radians(mag["angle"])
    mx = np.cos(angle_rad)
    my = np.sin(angle_rad)
    size = mag["size"]

    if mag["type"] == "Bar Magnet":
        width = size
        height = size * 0.3
        cx = x0 - mx * (size / 2)
        cy = y0 - my * (size / 2)

        rect = patches.Rectangle(
            (cx - width/2, cy - height/2), width, height,
            linewidth=1, edgecolor='white', facecolor='#444444', zorder=10
        )
        t = transforms.Affine2D().rotate_around(cx, cy, angle_rad) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
    else:
        radius = size * 0.2
        cx = x0 - mx * radius
        cy = y0 - my * radius

        circle = patches.Circle(
            (cx, cy), radius,
            fill=True, color='#444444', edgecolor='white', zorder=10
        )
        ax.add_patch(circle)

# AXIS SETTINGS
ax.set_xlim(-field_range, field_range)
ax.set_ylim(-field_range, field_range)
ax.set_aspect('equal')
ax.axis('off')

# LEGEND
handles, labels = ax.get_legend_handles_labels()
if labels:
    ax.legend(loc='upper right', framealpha=0.8, labelcolor='white', facecolor='black')

# RENDER
st.pyplot(fig)

# ============================================================
# INFO PANEL
# ============================================================
st.markdown("---")
st.markdown("""
## 🔬 TRUE 3D EQUILIBRIUM DETECTION ENGINE v2.0

본 실험실 시뮬레이터는 물리학적 신뢰성을 위해 다음 알고리즘이 내장되어 있습니다.

- **자석 내부 판정 마스킹**: 자석의 물리적 부피 내부 영역은 평형점 검출 대상에서 완전히 제외되어 수치적 특이점으로 인한 허상의 오류 점들을 원천 제거합니다.
- **구형 자석 물리식 고도화**: 구형 자석에 대해 이상적 자기 쌍극자 모멘트 수학 모델($\\mathbf{B} \\propto r^{-3}$)을 구현하여 사실적인 원거리-근거리 자계를 형성합니다.
- **유선(Streamline)의 착시 안내**: 화면에 표시되는 유선은 3D 입체 자기장을 선택하신 2D 평면($Z = z_{obs}$)에 투영한 결과물입니다. 유선이 특정 지점으로 모이거나 끊어지는 현상은 자기 법칙의 오류가 아닌, **자기력선이 Z축(화면 안팎) 방향으로 흐르고 있음**을 뜻하는 정상적인 물리 현상입니다.
""")