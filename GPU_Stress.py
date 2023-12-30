import moderngl_window as mglw
import sys

class App(mglw.WindowConfig):
    window_size = 1600, 900

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quad = mglw.geometry.quad_fs()

        shader_source = {
            'vertex_shader': '''
                #version 330 core

                in vec3 in_position;

                void main() {
                    gl_Position = vec4(in_position, 1.0);
                }
            ''',
            'fragment_shader': '''
                #version 330 core

                out vec4 fragColor;

                uniform vec2 u_resolution;
                uniform float u_time;
                uniform sampler2D u_texture1;
                uniform sampler2D u_texture2;
                uniform sampler2D u_texture3;

                const float PI = 3.1416;
                const float TAU = 2 * PI;


                float displace(vec3 p, sampler2D tex) {
                    float s = 4.5;
                    float u = s / TAU * atan(p.y / p.x);
                    float v = sign(p.z) / TAU *
                        acos((p.z * p.z * sqrt(s * s + 1) + sqrt(1 - p.z * p.z * s * s)) / (p.z * p.z + 1));
                    vec2 uv = 2.0 * vec2(u, v);
                    float disp = texture(tex, uv).r;
                    return disp * 0.06;
                }

                mat2 rot2D(float a) {
                    float sa = sin(a);
                    float ca = cos(a);
                    return mat2(ca, sa, -sa, ca);
                }

                void rotate(inout vec3 p) {
                    p.xy *= rot2D(sin(u_time * 0.8) * 0.25);
                    p.yz *= rot2D(sin(u_time * 0.7) * 0.2);
                }


                float map(vec3 p) {
                    float dist = length(vec2(length(p.xy) - 0.6, p.z)) - 0.22;
                    return dist * 0.7;
                }

                vec3 getNormal(vec3 p) {
                    vec2 e = vec2(0.01, 0.0);
                    vec3 n = vec3(map(p)) - vec3(map(p - e.xyy), map(p - e.yxy), map(p - e.yyx));
                    return normalize(n);
                }


                float rayMarch(vec3 ro, vec3 rd) {
                    float dist = 0.0;
                    for (int i = 0; i < 64; i++) {
                        vec3 p = ro + dist * rd;

                        rotate(p);
                        float hit = map(p);
                        dist += hit;

                        dist -= displace(0.5 * p, u_texture2);

                        if (dist > 100.0 || abs(hit) < 0.0001) break;
                    }
                    return dist;
                }


                vec3 triPlanar(sampler2D tex, vec3 p, vec3 normal) {
                    normal = abs(normal);
                    normal = pow(normal, vec3(15));
                    normal /= normal.x + normal.y + normal.z;
                    p = p * 0.5 + 0.5;
                    return (texture(tex, p.xy) * normal.z +
                            texture(tex, p.xz) * normal.y +
                            texture(tex, p.yz) * normal.x).rgb;
                }


                vec3 render(vec2 offset) {
                    vec2 uv = (2.0 * (gl_FragCoord.xy + offset) - u_resolution.xy) / u_resolution.y;
                    vec3 col = vec3(0);

                    vec3 ro = vec3(0, 0, -1.0);
                    vec3 rd = normalize(vec3(uv, 1.0));
                    float dist = rayMarch(ro, rd);

                    vec3 p = ro + dist * rd;
                    rotate(p);
                    col += triPlanar(u_texture1, p * 1.0, getNormal(p));
                    
                    return col;
                }


                vec3 renderAAx4() {
                    vec4 e = vec4(0.125, -0.125, 0.375, -0.375);
                    vec3 colAA = render(e.xz) + render(e.yw) + render(e.wx) + render(e.zy);
                    return colAA /= 4.0;
                }


                void main() {
                    vec3 color = renderAAx4();

                    fragColor = vec4(color, 1.0);
                }
            ''',
        }
        self.program = self.ctx.program(**shader_source)
        self.set_uniform('u_resolution', self.window_size)

    def set_uniform(self, u_name, u_value):
        try:
            self.program[u_name] = u_value
        except KeyError:
            print(f'{u_name} not used in shader')

    def render(self, time, frame_time):
        while True:
            self.ctx.clear()
            self.set_uniform('u_time', time)
            self.quad.render(self.program)

def gpu_stress_func():
    sys.argv.extend(['--window', 'headless'])
    mglw.run_window_config(App)