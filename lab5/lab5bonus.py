import taichi as ti

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
eta_glass = ti.field(ti.f32, shape=())
enable_aa = ti.field(ti.i32, shape=())

MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    """斯涅耳定律计算折射方向，返回(折射向量, 是否全反射)"""
    cos_theta_i = I.dot(N)
    sin2_theta_t = eta * eta * (1.0 - cos_theta_i * cos_theta_i)
    # 全反射判断
    is_total = False
    refracted_dir = ti.Vector([0.0, 0.0, 0.0])
    if sin2_theta_t > 1.0:
        is_total = True
    else:
        cos_theta_t = ti.sqrt(1.0 - sin2_theta_t)
        refracted_dir = eta * (I - cos_theta_i * N) - cos_theta_t * N
        refracted_dir = normalize(refracted_dir)
    return refracted_dir, is_total

@ti.func
def intersect_sphere(ro, rd, center, radius):
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
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 玻璃球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.95, 0.95, 1.0])
        hit_mat = MAT_GLASS

    # 镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 棋盘格地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.func
def compute_lighting(p, N, obj_color, light_pos):
    L = normalize(light_pos - p)
    shadow_ray_orig = p + N * 1e-4
    shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
    dist_to_light = (light_pos - p).norm()
    in_shadow = shadow_t < dist_to_light

    ambient = 0.2 * obj_color
    direct_light = ambient
    if not in_shadow:
        diff = ti.max(0.0, N.dot(L))
        diffuse = 0.8 * diff * obj_color
        direct_light += diffuse
    return direct_light

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])
    eta = eta_glass[None]
    samples_per_pixel = 4 if enable_aa[None] else 1

    for i, j in pixels:
        final_color = ti.Vector([0.0, 0.0, 0.0])

        for _ in range(samples_per_pixel):
            offset_u = 0.0
            offset_v = 0.0
            if enable_aa[None]:
                offset_u = ti.random() - 0.5
                offset_v = ti.random() - 0.5

            u = (i + offset_u - res_x / 2.0) / res_y * 2.0
            v = (j + offset_v - res_y / 2.0) / res_y * 2.0

            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            throughput = ti.Vector([1.0, 1.0, 1.0])
            pixel_color = ti.Vector([0.0, 0.0, 0.0])

            for bounce in range(max_bounces[None]):
                t, N, obj_color, mat_id = scene_intersect(ro, rd)

                if t > 1e9:
                    pixel_color += throughput * bg_color
                    break

                hit_point = ro + rd * t

                if mat_id == MAT_MIRROR:
                    ro = hit_point + N * 1e-4
                    rd = normalize(reflect(rd, N))
                    throughput *= 0.8 * obj_color

                elif mat_id == MAT_GLASS:
                    # 预先声明并初始化变量
                    normal = ti.Vector([0.0, 0.0, 0.0])
                    effective_eta = 1.0
                    
                    if rd.dot(N) < 0:
                        effective_eta = 1.0 / eta
                        normal = N
                    else:
                        effective_eta = eta
                        normal = -N

                    refracted_dir, is_total_reflection = refract(rd, normal, effective_eta)

                    if is_total_reflection:
                        ro = hit_point + N * 1e-4
                        rd = normalize(reflect(rd, N))
                    else:
                        ro = hit_point - normal * 1e-4
                        rd = refracted_dir
                    throughput *= 0.95 * obj_color

                elif mat_id == MAT_DIFFUSE:
                    direct_light = compute_lighting(hit_point, N, obj_color, light_pos)
                    pixel_color += throughput * direct_light
                    break

            final_color += pixel_color

        final_color /= samples_per_pixel
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing: Glass & Anti-Aliasing", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 3
    eta_glass[None] = 1.5
    enable_aa[None] = 1

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.3):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)
            eta_glass[None] = gui.slider_float('Refractive Index', eta_glass[None], 1.1, 2.5)
            enable_aa[None] = 1 if gui.checkbox('Anti-Aliasing (MSAA)', enable_aa[None]) else 0

        window.show()

if __name__ == '__main__':
    main()