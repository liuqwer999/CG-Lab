import taichi as ti
import math

# 初始化 Taichi，指定使用 CPU 后端
ti.init(arch=ti.cpu)

# --- 修改点1：声明 8 个顶点 (立方体) ---
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

# --- 修改点2：定义立方体的 12 条边 (顶点索引对) ---
# 这是一个 Python 列表，用于在绘制时告诉程序哪两个点相连
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0), # 后面 (z=-1)
    (4, 5), (5, 6), (6, 7), (7, 4), # 前面 (z=1)
    (0, 4), (1, 5), (2, 6), (3, 7)  # 连接前后的棱
]

@ti.func
def get_model_matrix(angle: ti.f32):
    """
    模型变换矩阵：为了体现3D效果，同时绕 Y轴 和 X轴 旋转
    """
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    
    # 1. 绕 Y 轴旋转矩阵 (左右转，看起来更有立体感)
    rot_y = ti.Matrix([
        [c,  0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 2. 绕 X 轴旋转矩阵 (稍微带一点俯仰)
    rot_x = ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s,  c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 3. 组合旋转：先绕X，再绕Y
    return rot_y @ rot_x

@ti.func
def get_view_matrix(eye_pos):
    """
    视图变换矩阵：将相机移动到原点
    """
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """
    透视投影矩阵 (保持不变，透视效果全靠它)
    """
    n = -zNear
    f = -zFar
    
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle: ti.f32):
    """
    计算顶点的坐标变换 (逻辑不变，只是顶点数变多了)
    """
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model
    
    # 循环从 3 次改成 8 次
    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    # --- 修改点3：初始化立方体的 8 个顶点 (中心在原点，边长为 2) ---
    # 后面 (z=-1)
    vertices[0] = [-1.0, -1.0, -1.0]
    vertices[1] = [ 1.0, -1.0, -1.0]
    vertices[2] = [ 1.0,  1.0, -1.0]
    vertices[3] = [-1.0,  1.0, -1.0]
    # 前面 (z=1)
    vertices[4] = [-1.0, -1.0,  1.0]
    vertices[5] = [ 1.0, -1.0,  1.0]
    vertices[6] = [ 1.0,  1.0,  1.0]
    vertices[7] = [-1.0,  1.0,  1.0]
    
    gui = ti.GUI("3D Cube (Bonus)", res=(700, 700))
    angle = 0.0
    
    print("Controls: Press A/D to rotate, ESC to exit")
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 5.0  # 按 A 顺时针转
            elif gui.event.key == 'd':
                angle -= 5.0  # 按 D 逆时针转
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        compute_transform(angle)
        
        # --- 修改点4：遍历 12 条边进行绘制 ---
        for i, j in cube_edges:
            a = screen_coords[i]
            b = screen_coords[j]
            # 用不同的颜色画，增加区分度 (这里统一用青色)
            gui.line(a, b, radius=2, color=0x00FFFF)
        
        gui.show()

if __name__ == '__main__':
    main()