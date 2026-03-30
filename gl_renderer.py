import OpenGL.GL as gl
import numpy as np
import glfw
import cv2

# Shaders
_VERT_QUAD = """#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTex;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTex;
}
"""

_FRAG_QUAD = """#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D frameTexture;
void main() {
    FragColor = texture(frameTexture, TexCoord);
}
"""

_VERT_FLAT = """#version 330 core
layout(location = 0) in vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

_FRAG_FLAT = """#version 330 core
out vec4 FragColor;
uniform vec4 color;
void main() {
    FragColor = color;
}
"""

class GLRenderer:
    def __init__(self, width, height, title="GL Window"):
        self.width = width
        self.height = height
        self.window = self._init_glfw(title)
        self._init_gl()
        
    def _init_glfw(self, title):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        
        window = glfw.create_window(self.width, self.height, title, None, None)
        if not window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")
            
        glfw.make_context_current(window)
        return window

    def _compile_shader(self, vsrc, fsrc):
        from OpenGL.GL.shaders import compileProgram, compileShader
        return compileProgram(
            compileShader(vsrc, gl.GL_VERTEX_SHADER), 
            compileShader(fsrc, gl.GL_FRAGMENT_SHADER)
        )

    def _init_gl(self):
        self.quad_shader = self._compile_shader(_VERT_QUAD, _FRAG_QUAD)
        self.flat_shader = self._compile_shader(_VERT_FLAT, _FRAG_FLAT)

        self.quad_vao = gl.glGenVertexArrays(1)
        self.quad_vbo = gl.glGenBuffers(1)
        self.quad_ebo = gl.glGenBuffers(1)

        vertices = np.array([
            -1.0,  1.0,  0.0, 0.0, 
             1.0,  1.0,  1.0, 0.0, 
             1.0, -1.0,  1.0, 1.0, 
            -1.0, -1.0,  0.0, 1.0  
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        gl.glBindVertexArray(self.quad_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.quad_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.quad_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, gl.ctypes.c_void_p(8))
        gl.glEnableVertexAttribArray(1)

        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.width, self.height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        self.line_vao = gl.glGenVertexArrays(1)
        self.line_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.line_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.line_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 8192 * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 2 * 4, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def _pixel_to_ndc(self, px, py):
        nx = (px / self.width) * 2.0 - 1.0
        ny = 1.0 - (py / self.height) * 2.0
        return nx, ny

    def _upload_frame(self, frame):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb_frame.tobytes())

    def _draw_quad(self):
        gl.glUseProgram(self.quad_shader)
        gl.glBindVertexArray(self.quad_vao)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)

    def _draw_boxes(self, detections):
        gl.glUseProgram(self.flat_shader)
        gl.glBindVertexArray(self.line_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.line_vbo)

        color_loc = gl.glGetUniformLocation(self.flat_shader, "color")

        for i, det in enumerate(detections):
            bb = det.bounding_box
            if det.detection_type == "face":
                color = (0.0, 1.0, 0.31, 1.0)
            else:
                from detector import get_color
                c = get_color(i)
                color = (c[2]/255.0, c[1]/255.0, c[0]/255.0, 1.0)
            
            gl.glUniform4f(color_loc, *color)

            x1, y1 = bb.x, bb.y
            x2, y2 = bb.x + bb.width, bb.y + bb.height

            n_x1, n_y1 = self._pixel_to_ndc(x1, y1)
            n_x2, n_y2 = self._pixel_to_ndc(x2, y2)

            # Main bounding box
            verts = np.array([
                n_x1, n_y1,
                n_x2, n_y1,
                n_x2, n_y2,
                n_x1, n_y2
            ], dtype=np.float32)
            
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
            gl.glDrawArrays(gl.GL_LINE_LOOP, 0, 4)

            # Corner accents
            lw = (n_x2 - n_x1) * 0.15
            lh = (n_y1 - n_y2) * 0.15 

            accent_verts = np.array([
                n_x1, n_y1, n_x1 + lw, n_y1,
                n_x1, n_y1, n_x1, n_y1 - lh,
                n_x2, n_y1, n_x2 - lw, n_y1,
                n_x2, n_y1, n_x2, n_y1 - lh,
                n_x1, n_y2, n_x1 + lw, n_y2,
                n_x1, n_y2, n_x1, n_y2 + lh,
                n_x2, n_y2, n_x2 - lw, n_y2,
                n_x2, n_y2, n_x2, n_y2 + lh,
            ], dtype=np.float32)
            
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, accent_verts.nbytes, accent_verts)
            gl.glLineWidth(3.0)
            gl.glDrawArrays(gl.GL_LINES, 0, 16)
            gl.glLineWidth(1.0)

            # Skeleton drawing
            if hasattr(det, 'keypoints') and det.keypoints:
                SKELETON_EDGES = [
                    (15, 13), (13, 11), (11, 12), (12, 14), (14, 16),
                    (11, 5), (12, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
                ]
                sk_verts = []
                for pt1, pt2 in SKELETON_EDGES:
                    if pt1 < len(det.keypoints) and pt2 < len(det.keypoints):
                        k1 = det.keypoints[pt1]
                        k2 = det.keypoints[pt2]
                        if k1.confidence > 0.5 and k2.confidence > 0.5:
                            kx1, ky1 = self._pixel_to_ndc(k1.x, k1.y)
                            kx2, ky2 = self._pixel_to_ndc(k2.x, k2.y)
                            sk_verts.extend([kx1, ky1, kx2, ky2])
                
                if sk_verts:
                    sk_arr = np.array(sk_verts, dtype=np.float32)
                    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, sk_arr.nbytes, sk_arr)
                    gl.glLineWidth(2.0)
                    gl.glDrawArrays(gl.GL_LINES, 0, len(sk_arr) // 2)
                    gl.glLineWidth(1.0)

    def render(self, frame, detections):
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        self._upload_frame(frame)
        self._draw_quad()
        
        if detections:
            self._draw_boxes(detections)
            
        glfw.swap_buffers(self.window)
