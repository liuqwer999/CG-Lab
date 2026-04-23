import taichi as ti

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())
enable_shadow = ti.field(ti.i32, shape=())

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

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
def intersect_cone(ro, rd, apex, base_y, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2
    ro_local = ro - apex
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2

    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            # 修复：大写 B 而非小写 b
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)
            t_first = min(t1, t2)
            t_second = max(t1, t2)
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

@ti.func
def intersect_plane(ro, rd, plane_y, plane_normal):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t_candidate = (plane_y - ro.y) / rd.y
        if t_candidate > 0:
            t = t_candidate
            normal = plane_normal
    return t, normal

@ti.func
def is_in_shadow(p, light_pos):
    shadow_ray_dir = normalize(light_pos - p)
    shadow_ray_origin = p + shadow_ray_dir * 1e-4
    light_distance = (light_pos - p).norm()
    t_sph, _ = intersect_sphere(shadow_ray_origin, shadow_ray_dir, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
    t_cone, _ = intersect_cone(shadow_ray_origin, shadow_ray_dir, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
    return (t_sph > 0 and t_sph < light_distance) or (t_cone > 0 and t_cone < light_distance)

@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])

        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])

        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])

        t_plane, n_plane = intersect_plane(ro, rd, -1.5, ti.Vector([0.0, 1.0, 0.0]))
        if 0 < t_plane < min_t:
            min_t = t_plane
            hit_normal = n_plane
            p = ro + rd * t_plane
            checker = (ti.floor(p.x * 2) + ti.floor(p.z * 2)) % 2
            hit_color = ti.Vector([0.3, 0.3, 0.3]) if checker == 0 else ti.Vector([0.5, 0.5, 0.5])

        color = ti.Vector([0.05, 0.15, 0.15])

        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])
            L = normalize(light_pos - p)
            V = normalize(ro - p)

            shadow_factor = 1.0
            if enable_shadow[None] == 1 and is_in_shadow(p, light_pos):
                shadow_factor = 0.2

            ambient = Ka[None] * light_color * hit_color
            diffuse = Kd[None] * ti.max(0.0, N.dot(L)) * light_color * hit_color * shadow_factor

            # Blinn-Phong 高光计算
            H = normalize(L + V)
            spec = ti.max(0.0, N.dot(H))
            specular = Ks[None] * (spec ** shininess[None]) * light_color * shadow_factor

            color = ambient + diffuse + specular
            pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Blinn-Phong + Hard Shadow", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    # 初始参数
    Ka[None] = 0.18
    Kd[None] = 0.72
    Ks[None] = 0.48
    shininess[None] = 36.0
    enable_shadow[None] = 0

    while window.running:
        render()
        canvas.set_image(pixels)
        # UI 完全保留原版顺序
        with gui.sub_window("Material Parameters", 0.7, 0.05, 0.28, 0.55):
            Ka[None] = gui.slider_float('Ka (Ambient)', Ka[None], 0.0, 1.0)
            gui.text("")
            Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
            gui.text("")
            Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
            gui.text("")
            shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 128.0)
            gui.text("")
            enable_shadow[None] = gui.checkbox("Enable Hard Shadow", enable_shadow[None])
        window.show()

if __name__ == '__main__':
    main()