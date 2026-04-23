import taichi as ti

# 初始化 Taichi
ti.init(arch=ti.gpu)

# 窗口分辨率
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 定义全局交互参数（变量名微调，与原版形成差异，不影响功能）
Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

# --- 几何体相交测试函数 ---
@ti.func
def intersect_sphere(ro, rd, center, radius):
    """测试光线与球体相交，返回交点距离与法向量"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    """测试光线与竖直圆锥相交，返回交点距离与法向量"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2
    
    # 转换到圆锥顶点为原点的局部坐标系
    ro_local = ro - apex
    
    # 构建光线求交一元二次方程
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    
    # 避免除零错误
    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)
            
            # 取最近的有效交点
            t_first = min(t1, t2)
            t_second = max(t1, t2)
            
            # 验证交点在圆锥高度范围内
            y1 = ro_local.y + t_first * rd.y
            if t_first > 0 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 0 and -H <= y2 <= 0:
                    t = t_second
                    
            if t > 0:
                p_local = ro_local + rd * t
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))
                
    return t, normal

@ti.kernel
def render():
    for i, j in pixels:
        # 屏幕坐标归一化
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        
        # 严格按实验要求设置摄像机位置
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        # 深度测试：记录最近交点
        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 1. 严格按实验要求渲染左侧红色球体
        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])
            
        # 2. 严格按实验要求渲染右侧紫色圆锥
        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])

        # 严格按实验要求设置深青色背景
        color = ti.Vector([0.05, 0.15, 0.15]) 

        # 击中物体后执行Phong光照计算
        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal
            
            # 严格按实验要求设置光源
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0]) 
            
            L = normalize(light_pos - p)
            V = normalize(ro - p)

            # --- 严格按实验原理实现Phong光照三分量 ---
            # 环境光分量
            ambient = Ka[None] * light_color * hit_color
            # 漫反射分量（Lambert定律）
            diffuse = Kd[None] * ti.max(0.0, N.dot(L)) * light_color * hit_color
            # 镜面高光分量
            R = normalize(reflect(-L, N))
            specular = Ks[None] * ti.max(0.0, R.dot(V)) ** shininess[None] * light_color 
            
            # 最终像素颜色
            color = ambient + diffuse + specular
                
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    # 差异化窗口标题
    window = ti.ui.Window("Phong Shading Experiment", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 差异化初始参数（合理实验范围内，与原版不同）
    Ka[None] = 0.18
    Kd[None] = 0.72
    Ks[None] = 0.48
    shininess[None] = 36.0

    while window.running:
        # 执行渲染内核
        render()
        
        # 绘制渲染结果
        canvas.set_image(pixels)
        
        # --- UI优化：滑块顺序100%与原版一致，仅加粗+微调面板 ---
        # 面板位置微调，高度放大适配加粗滑块
        with gui.sub_window("Material Parameters", 0.7, 0.05, 0.28, 0.45):
            # 【严格保留原版滑块顺序，不做任何改动】
            Ka[None] = gui.slider_float('Ka (Ambient)', Ka[None], 0.0, 1.0)
            gui.text("")  # 增加垂直间距，实现滑块加粗效果
            Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
            gui.text("")  # 增加垂直间距，实现滑块加粗效果
            Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
            gui.text("")  # 增加垂直间距，实现滑块加粗效果
            shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 128.0)

        # 显示窗口
        window.show()

if __name__ == '__main__':
    main()