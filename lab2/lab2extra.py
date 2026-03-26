import taichi as ti
import math

# 初始化
ti.init(arch=ti.cpu)

# 窗口参数
width, height = 800, 800

# -------------------------- 数据定义 --------------------------
# 8个顶点
cube_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
# 12条边 (用两个一维场代替二维场，彻底避免索引问题)
edge_start = ti.field(dtype=ti.i32, shape=12)
edge_end = ti.field(dtype=ti.i32, shape=12)
# 用于渲染的屏幕坐标
screen_points = ti.Vector.field(2, dtype=ti.f32, shape=8)

# -------------------------- 初始化数据 (纯Python，不报错) --------------------------
# 顶点数据
v_data = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
]
# 边数据
e_start_data = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
e_end_data   = [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7]

# 复制到Taichi field
for i in range(8):
    cube_vertices[i] = ti.Vector(v_data[i])
for i in range(12):
    edge_start[i] = e_start_data[i]
    edge_end[i] = e_end_data[i]

# -------------------------- 渲染核心 --------------------------
@ti.kernel
def render(angle_x: ti.f32, angle_y: ti.f32):
    # 1. 计算旋转矩阵 (直接在Kernel里算，不用ti.func)
    cx = ti.cos(angle_x)
    sx = ti.sin(angle_x)
    cy = ti.cos(angle_y)
    sy = ti.sin(angle_y)
    
    # 模型矩阵 (Model)
    model = ti.Matrix([
        [cy, 0, sy, 0],
        [sx*sy, cx, -sx*cy, 0],
        [-cx*sy, sx, cx*cy, 0],
        [0, 0, 0, 1]
    ])
    
    # 视图矩阵 (View)：相机在 (0, 0, 4)
    view = ti.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -4],
        [0, 0, 0, 1]
    ])
    
    # 投影矩阵 (Projection)：简单透视
    fov = 60.0 * 3.1415926 / 180.0
    aspect = width / height
    near = 0.1
    far = 100.0
    proj = ti.Matrix.zero(ti.f32, 4, 4)
    proj[0, 0] = 1.0 / (aspect * ti.tan(fov/2))
    proj[1, 1] = 1.0 / ti.tan(fov/2)
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    
    # 组合 MVP
    mvp = proj @ view @ model
    
    # 2. 变换顶点
    for i in range(8):
        v = ti.Vector([cube_vertices[i][0], cube_vertices[i][1], cube_vertices[i][2], 1.0])
        v_clip = mvp @ v
        v_ndc = v_clip / v_clip[3]
        # 转屏幕坐标
        screen_points[i][0] = (v_ndc[0] + 1.0) * 0.5
        screen_points[i][1] = (v_ndc[1] + 1.0) * 0.5

# -------------------------- 主循环 --------------------------
gui = ti.GUI("3D Cube", res=(width, height))

# --- 修改部分开始 ---
t = 0.0
speed = 0.005  # 这里的speed现在是一个可正可负的变量
# --- 修改部分结束 ---

while gui.running:
    # --- 修改部分开始 ---
    # 插值逻辑：平滑往返 (0 -> 1 -> 0)
    t += speed
    
    # 碰到边界就反弹（速度反向）
    if t >= 1.0:
        t = 1.0
        speed = -speed  # 开始往回走
    elif t <= 0.0:
        t = 0.0
        speed = -speed  # 开始往前走
    # --- 修改部分结束 ---
        
    # 定义姿态 R0 (t=0) 和 R1 (t=1)
    # R0: 向左转45度，向下看20度
    rx0 = -20.0 * math.pi / 180.0
    ry0 = -45.0 * math.pi / 180.0
    
    # R1: 向右转45度，向上看20度
    rx1 = 20.0 * math.pi / 180.0
    ry1 = 45.0 * math.pi / 180.0
    
    # 简单线性插值角度
    current_rx = rx0 * (1 - t) + rx1 * t
    current_ry = ry0 * (1 - t) + ry1 * t
    
    # 渲染
    render(current_rx, current_ry)
    
    # 画线
    for i in range(12):
        s = screen_points[edge_start[i]]
        e = screen_points[edge_end[i]]
        gui.line(s, e, color=0x4CC9F0, radius=2)
    
    gui.text(f"t = {t:.2f}", (0.05, 0.95), color=0xFFFFFF)
    gui.show()