import taichi as ti

# 注意：初始化必须在最前面执行，接管底层 GPU
ti.init(arch=ti.gpu)

# 添加 src 目录到 Python 搜索路径，解决子模块导入
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# 导入 Work0 模块的核心内容（和你 src/Work0/main.py 导入逻辑一致）
from Work0.config import WINDOW_RES, PARTICLE_COLOR, PARTICLE_RADIUS
from Work0.physics import init_particles, update_particles, pos

def main():
    print("正在编译 GPU 内核，请稍候...")
    init_particles()
    
    gui = ti.GUI("Experiment 0: Taichi Gravity Swarm", res=WINDOW_RES)
    print("编译完成！请在弹出的窗口中移动鼠标。")
    
    # 渲染主循环（和你 src/Work0/main.py 完全一致）
    while gui.running:
        mouse_x, mouse_y = gui.get_cursor_pos()
        
        # 驱动 GPU 进行物理计算
        update_particles(mouse_x, mouse_y)
        
        # 读取显存数据并绘制
        gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
        gui.show()

if __name__ == "__main__":
    main()