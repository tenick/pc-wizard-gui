import moderngl_window as mglw
import sys
import numpy as np

class App(mglw.WindowConfig):
    window_size = 15000, 15000
    #resource_dir = 'resources'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.quad = mglw.geometry.quad_fs()
        self.program = self.ctx.program(vertex_shader="""
        #version 330
        in vec2 iv;
        in vec3 ic;
        out vec3 color;
        void main() {
            gl_Position = vec4(iv, 0.0, 1.0);
            color = ic;
        }
        """,
        fragment_shader="""
        #version 330
        out vec4 fragColor;
        in vec3 color;
        void main() {
            fragColor = vec4(color, 1.0);
        }
        """,
        )
        vertices = np.array([
            -1.0,  -1.0,   1.0, 0.0, 0.0,
            1.0,  -1.0,   0.0, 1.0, 0.0,
            0.0,   1.0,   0.0, 0.0, 1.0],
            dtype='f4',
        )
        vertices = np.repeat(1, 1000)
        self.vao = self.ctx.simple_vertex_array(self.program, self.ctx.buffer(vertices), 'iv', 'ic')

    def render(self, time, frame_time):
            #self.quad.render(self.program)
        #while True:
        self.vao.render()
        self.ctx.clear(1, 1, 1, 1)

def gpu_stress_func():
    sys.argv.extend(['--window', 'headless'])
    mglw.run_window_config(App)