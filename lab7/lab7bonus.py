import taichi as ti
import math

# 初始化 Taichi，使用 GPU 加速
ti.init(arch=ti.gpu)

# ================= 物理与网格参数 =================
N = 20                  # 布料网格分辨率 N x N
mass = 1.0              # 质点质量
dt = 5e-4               # 时间步长
k_s = 10000.0           # 弹簧劲度系数
k_d = 1.0               # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

# 碰撞参数
restitution = 0.3       # 恢复系数 (0~1)
sphere_center = ti.Vector.field(3, dtype=float, shape=())
sphere_radius = ti.field(dtype=float, shape=())

# ================= Taichi 数据场 =================
x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

# 弹簧数据：最大数量需容纳结构+剪切+弯曲弹簧
max_springs = N * N * 8   # 足够大
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())

# 球体渲染顶点（局部坐标）和世界坐标
sphere_vertices = ti.Vector.field(3, dtype=float, shape=600)
sphere_world_pos = ti.Vector.field(3, dtype=float, shape=600)

# ================= 初始化 Kernels =================
@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        # 固定角点
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        # 1. 结构弹簧 (Structural)
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()

        # 2. 剪切弹簧 (Shear)
        if i < N - 1 and j < N - 1:
            idx_diag = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag])
            spring_lengths[c] = (x[idx] - x[idx_diag]).norm()
        if i < N - 1 and j > 0:
            idx_diag2 = (i + 1) * N + (j - 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag2])
            spring_lengths[c] = (x[idx] - x[idx_diag2]).norm()

        # 3. 弯曲弹簧 (Bending)
        if i < N - 2:
            idx_bend = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_bend])
            spring_lengths[c] = (x[idx] - x[idx_bend]).norm()
        if j < N - 2:
            idx_bend2 = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_bend2])
            spring_lengths[c] = (x[idx] - x[idx_bend2]).norm()

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

@ti.kernel
def init_sphere_vertices():
    """在球面上均匀采样，用于渲染球体"""
    n = 600
    for i in range(n):
        # 使用 Fibonacci 球体分布
        y = 1.0 - (2.0 * i + 1.0) / n
        radius_at_y = ti.sqrt(1.0 - y * y)
        theta = 2.399963 * i  # 黄金角比例
        sx = ti.cos(theta) * radius_at_y
        sz = ti.sin(theta) * radius_at_y
        sphere_vertices[i] = ti.Vector([sx, y, sz])

@ti.kernel
def update_sphere_world_pos():
    """根据球体中心和半径更新世界坐标，用于渲染"""
    for i in range(600):
        sphere_world_pos[i] = sphere_vertices[i] * sphere_radius[None] + sphere_center[None]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()
    # 设置球体初始位置和半径
    sphere_center[None] = ti.Vector([0.0, 0.3, 0.0])
    sphere_radius[None] = 0.25
    init_sphere_vertices()
    update_sphere_world_pos()   # 第一次计算世界坐标

# ================= 力与碰撞处理 =================
@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    # 清空力，加重力与阻尼
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]

    # 弹簧力（原子加）
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            f_spring = -k_s * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.func
def resolve_sphere_collision(pos: ti.template(), vel: ti.template()):
    """球体碰撞响应：将进入球内的质点推出，并调整速度"""
    for i in range(N * N):
        if is_fixed[i] == 0:
            d = pos[i] - sphere_center[None]
            dist = d.norm()
            if dist < sphere_radius[None]:
                normal = d / dist
                # 推至球面
                pos[i] = sphere_center[None] + normal * sphere_radius[None]
                # 速度响应：移除指向球心的分量
                vn = vel[i].dot(normal)
                if vn < 0:
                    vel[i] -= (1.0 + restitution) * vn * normal

# ================= 积分 Kernels =================
@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
    resolve_sphere_collision(x, v)

@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
    resolve_sphere_collision(x, v)

@ti.kernel
def step_implicit_iter():
    # 1. 复制当前状态
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]

    # 2. 定点迭代（编译期展开）
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt

    # 3. 写回
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

    resolve_sphere_collision(x, v)

# ================= 主循环 =================
def main():
    init_cloth()

    window = ti.ui.Window("Cloth with Sphere Collision", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1  # 0: 显式, 1: 半隐式, 2: 隐式
    paused = False

    while window.running:
        # ----- GUI 面板 -----
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.36)
        window.GUI.text("Integration Method:")

        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler"):
            current_method = 0; init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler"):
            current_method = 1; init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler"):
            current_method = 2; init_cloth()

        window.GUI.text("")
        pause_label = "Resume" if paused else "Pause"
        if window.GUI.button(pause_label):
            paused = not paused
        if window.GUI.button("Reset Cloth"):
            init_cloth()
        window.GUI.end()

        # ----- 物理更新 -----
        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        # ----- 渲染 -----
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        # 布料
        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))

        # 球体：先更新世界坐标，再渲染
        update_sphere_world_pos()
        scene.particles(sphere_world_pos, radius=0.012, color=(1.0, 0.5, 0.0))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()