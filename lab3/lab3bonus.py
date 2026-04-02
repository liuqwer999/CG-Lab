import taichi as ti
import numpy as np

# 使用 gpu 后端
ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000  # 曲线采样点数量
POINT_RADIUS = 0.006  # 控制点半径，用于点击检测

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# GUI 绘制数据缓冲池
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# 用于存放曲线坐标的 GPU 缓冲区
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

# 功能状态控制
current_mode = 'bezier'  # 'bezier' 或 'bspline'
antialiasing = False     # 反走样开关

# 拖拽状态变量
dragging = False
drag_index = -1  # 当前正在拖拽的控制点索引

def de_casteljau(points, t):
    """纯 Python 递归实现 De Casteljau 算法（贝塞尔曲线）"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

def uniform_cubic_bspline(control_points, t):
    """均匀三次 B 样条曲线计算（矩阵形式）"""
    M = np.array([
        [-1/6,  3/6, -3/6, 1/6],
        [ 3/6, -6/6,  3/6, 0/6],
        [-3/6,  0/6,  3/6, 0/6],
        [ 1/6,  4/6,  1/6, 0/6]
    ])
    
    T = np.array([t**3, t**2, t, 1])
    P = np.array(control_points)
    
    return T @ M @ P

def compute_bspline_curve(control_points):
    """计算完整的 B 样条曲线（分段拼接）"""
    curve_points = []
    n = len(control_points)
    if n < 4:
        return curve_points
    
    for i in range(n - 3):
        segment_points = control_points[i:i+4]
        for t_int in range(NUM_SEGMENTS // (n - 3) + 1):
            t = t_int / (NUM_SEGMENTS // (n - 3))
            pt = uniform_cubic_bspline(segment_points, t)
            curve_points.append(pt)
    
    return curve_points

@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32, antialiasing: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        x_float = pt[0] * WIDTH
        y_float = pt[1] * HEIGHT
        
        if antialiasing == 1:
            # 反走样：3x3 邻域距离衰减
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x_pixel = ti.cast(x_float, ti.i32) + dx
                    y_pixel = ti.cast(y_float, ti.i32) + dy
                    
                    if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
                        dist = ti.sqrt((x_pixel + 0.5 - x_float)**2 + (y_pixel + 0.5 - y_float)**2)
                        weight = ti.exp(-dist**2 / 0.5)
                        current_color = pixels[x_pixel, y_pixel]
                        new_color = ti.Vector([0.0, 1.0, 0.0])
                        pixels[x_pixel, y_pixel] = current_color + new_color * weight
        else:
            # 普通绘制
            x_pixel = ti.cast(x_float, ti.i32)
            y_pixel = ti.cast(y_float, ti.i32)
            if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
                pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])

def get_point_under_cursor(cursor_pos, control_points):
    """返回鼠标下的控制点索引，没有则返回-1"""
    for i, point in enumerate(control_points):
        dx = cursor_pos[0] - point[0]
        dy = cursor_pos[1] - point[1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < POINT_RADIUS * 2:
            return i
    return -1

def main():
    global current_mode, antialiasing, dragging, drag_index
    
    window = ti.ui.Window("Bezier Curve + B-Spline + Antialiasing", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    
    print("=== 完整操作说明 ===")
    print("鼠标左键点击：添加控制点")
    print("鼠标左键按住控制点拖拽：移动控制点")
    print("Backspace 键：撤销上一个点")
    print("C 键：清空画布")
    print("B 键：切换贝塞尔/B样条模式")
    print("A 键：开关反走样")
    
    while window.running:
        cursor_pos = window.get_cursor_pos()
        
        # 拖拽逻辑
        if dragging and drag_index != -1:
            control_points[drag_index] = cursor_pos
        
        # ✅ 修复：分开处理 PRESS 和 RELEASE 事件
        # 处理 PRESS 事件
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB: 
                idx = get_point_under_cursor(cursor_pos, control_points)
                if idx != -1:
                    dragging = True
                    drag_index = idx
                    print(f"开始拖拽控制点 {idx}")
                else:
                    if len(control_points) < MAX_CONTROL_POINTS:
                        control_points.append(cursor_pos)
                        print(f"Added control point: {cursor_pos} (Total: {len(control_points)})")
            elif e.key == 'c': 
                control_points = []
                dragging = False
                drag_index = -1
                print("Canvas cleared.")
            elif e.key == 'b':
                current_mode = 'bspline' if current_mode == 'bezier' else 'bezier'
                print(f"Switched to {current_mode.upper()} mode")
            elif e.key == 'a':
                antialiasing = not antialiasing
                print(f"Antialiasing: {'ON' if antialiasing else 'OFF'}")
            elif e.key == ti.ui.BACKSPACE:
                if control_points:
                    removed_point = control_points.pop()
                    print(f"撤销上一个点: {removed_point} (剩余: {len(control_points)})")
                    dragging = False
                    drag_index = -1
        
        # 处理 RELEASE 事件
        for e in window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB and dragging:
                dragging = False
                drag_index = -1
                print("结束拖拽")
        
        clear_pixels()
        
        current_count = len(control_points)
        if (current_mode == 'bezier' and current_count >= 2) or (current_mode == 'bspline' and current_count >= 4):
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            
            if current_mode == 'bezier':
                for t_int in range(NUM_SEGMENTS + 1):
                    t = t_int / NUM_SEGMENTS
                    curve_points_np[t_int] = de_casteljau(control_points, t)
            else:
                bspline_pts = compute_bspline_curve(control_points)
                if bspline_pts:
                    indices = np.linspace(0, len(bspline_pts)-1, NUM_SEGMENTS + 1, dtype=int)
                    curve_points_np = np.array(bspline_pts)[indices]
            
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(NUM_SEGMENTS + 1, 1 if antialiasing else 0)
                    
        canvas.set_image(pixels)
        
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=POINT_RADIUS, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))
        
        window.show()

if __name__ == '__main__':
    main()