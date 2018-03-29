"""
Environment for Robot Arm.
You can customize this script in a way you want.

View more on : https://morvanzhou.github.io/tutorials/


Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""
import numpy as np
import pyglet


pyglet.clock.set_fps_limit(10000)


# rectangle class
# https://stackoverflow.com/questions/26808513/drawing-a-rectangle-around-mouse-drag-pyglet
class Rect:
    def __init__(self, x, y, w, h, c):
        self.set(x, y, w, h)
        self.set_color(c)

        
    def draw(self):
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, self._quad , ('c3B', self._color))

    def set_color(self, c):
        self._color = c

    def set(self, x=None, y=None, w=None, h=None):
        self._x = self._x if x is None else x
        self._y = self._y if y is None else y
        self._w = self._w if w is None else w
        self._h = self._h if h is None else h
        half_w = self._w / 2
        half_h = self._h / 2

        self._quad = ('v2f',   (self._x - half_w, self._y - half_h,
                                self._x + half_w, self._y - half_h,
                                self._x + half_w, self._y + half_h,
                                self._x - half_w, self._y + half_h))

class Viewer(pyglet.window.Window):
    color = {
        'background': [0]*3 + [0]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    frame_count = 0
    output_dir = 'rawpic'

    def __init__(self, width, height, target_rect_pos, rect_w, rect_h, predict_rect_pos, mouse_in):
        super(Viewer, self).__init__(width, height, resizable=False, caption='Arm', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.target_rect_pos = target_rect_pos
        self.predict_rect_pos = predict_rect_pos
        self.mouse_in = mouse_in
        self.rect_w = rect_w

        self.center_coord = np.array((min(width, height)/2, ) * 2)
        # self.batch = pyglet.graphics.Batch()

        c1, c2, c3 = (255, 0, 0)*4, (255, 0, 0)*4, (0, 255, 0)*4
        # self.target_rect = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', point_box), ('c3B', c2))
        self.target_rect = Rect(target_rect_pos[0], target_rect_pos[1], rect_w,rect_h, c2 )
        self.predict_rect = Rect(predict_rect_pos[0], predict_rect_pos[1], rect_w,rect_h, c3 )
        
        # Set up the two top labels
        # self.target_label = pyglet.text.Label(text="Target (Red): ", x=10, y=height -20)
        # self.predict_label = pyglet.text.Label(text="Predict (Red): ", x=110, y=height -20)
        self.target_label = pyglet.text.Label(text="Target (Red): ", x=10, y=height -20, font_size=8.0)
        self.predict_label = pyglet.text.Label(text="Predict (Blue): ", x=10, y=height -40 , font_size=8.0)


        self.pos_list = []

    def render(self):
        pyglet.clock.tick()
        self._update_rect()
        # self.switch_to()  #only one window
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()


    # def save_a_frame(self):
    #     file_num=str(__class__.frame_count).zfill(5)
    #     # file_t=str(self.chrono)
    #     filename="pic/frame-"+file_num+'.png'
    #     print
    #     pyglet.image.get_buffer_manager().get_color_buffer().save(filename)
    #     print('image file writen : ',filename)

    #     __class__.frame_count = __class__.frame_count + 1

    def save_a_frame(self):
        file_num=str(__class__.frame_count).zfill(5)
        # file_t=str(self.chrono)
        filename= self.output_dir + "/frame-"+file_num+'.png'
        # color_buf = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().get_data("RGB",640*3)
        
        # # color_buf = pyglet.image.get_image_data().get_data("RGB",640)
    
        # print('type(color_buf)=' + str(type(color_buf)))
        # print('len(color_buf)=' + str(len(color_buf)))
        # print('color_buf[0]=' + str(color_buf[0]))
        
        # np_color_buf = np.array(color_buf)
        # print("np_color_buf.shape = " +  str(np_color_buf.shape) )

        pyglet.image.get_buffer_manager().get_color_buffer().save(filename)
        print('image file writen : ',filename)
        # print('__class__.frame_count = ', __class__.frame_count)
        # print('self.frame_count = ', self.frame_count)
        __class__.frame_count = __class__.frame_count + 1

    def reset_count_post_list(self):
       __class__.frame_count = 0
       self.pos_list = []

    def save_pos_list(self):
        np.savetxt(self.output_dir + '/_pos.out', self.pos_list, fmt='%3d') 

    def set_ouput_dir(self, d):
        self.output_dir = d

    def on_draw(self):
        self.clear()
        self.target_rect.draw()
        
        # self.target_label.draw()
        self.save_a_frame()
        self.pos_list.append(self.target_rect_pos.copy())

        self.predict_rect.draw()
        self.target_label.draw()
        self.predict_label.draw()
 

    def _update_rect(self):
        self.target_rect.set(x = self.target_rect_pos[0], y = self.target_rect_pos[1])
        self.predict_rect.set(x = self.predict_rect_pos[0], y = self.predict_rect_pos[1])

        self.target_label.text = "Target (Red): " + str(self.target_rect_pos)
        self.predict_label.text = "Predict (Blue): " + str(self.target_rect_pos)


    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            pyglet.clock.set_fps_limit(1000)
        elif symbol == pyglet.window.key.A:
            pyglet.clock.set_fps_limit(30)

    def on_mouse_motion(self, x, y, dx, dy):
        self.target_rect_pos[:] = [x, y]

    def on_mouse_enter(self, x, y):
        self.mouse_in[0] = True

    def on_mouse_leave(self, x, y):
        self.mouse_in[0] = False



class RectEnv(object):
    viewer = None
    
    get_point = False
    mouse_in = np.array([False])

    # viewer_xy = (640 , 480)
    # target_rect_w = 80
    # target_rect_h = 50

    viewer_xy = (240 , 240)
    target_rect_w = 30
    target_rect_h = 20

    def __init__(self):
        self.target_rect_pos = np.array([self.viewer_xy[0] * 0.5, self.viewer_xy[1]* 0.5])
        self.predict_rect_pos = np.array([self.viewer_xy[0]/2, self.viewer_xy[1]* 0.5])
        self.center_coord = np.array(self.viewer_xy)/2
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.target_rect_pos, self.target_rect_w, self.target_rect_h, self.predict_rect_pos, self.mouse_in)
        
        self.render() # ignore first pic

    def render(self, save_frame = False, predict_rect_pos = None):
            
        self.viewer.render()

    def get_random_rect_pos(self):
        x = np.clip(np.random.rand(1) * self.viewer_xy[0], 0 + self.target_rect_w, self.viewer_xy[0] - self.target_rect_w)
        y = np.clip(np.random.rand(1) * self.viewer_xy[1], 0 + self.target_rect_h, self.viewer_xy[1] - self.target_rect_h)
            
        return [int(x), int(y)]

    def random_rect_pos(self):
        if not self.mouse_in:
            self.target_rect_pos[:] = self.get_random_rect_pos()

        self.predict_rect_pos[:] = self.get_random_rect_pos()


    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def save_pos_list(self):
        self.viewer.save_pos_list()


    def recreate_dir(self, tar_dir):
        from shutil import rmtree
        import os
        if os.path.isdir(tar_dir):
            rmtree(tar_dir)
        os.mkdir(tar_dir)

    def main(self, output_dir = 'train_data', num = 1000):
        
        self.viewer.reset_count_post_list()
        self.recreate_dir(output_dir)
        self.viewer.set_ouput_dir(output_dir)
        for i in range(num):
            self.random_rect_pos()
            self.render()


        self.save_pos_list()



if __name__ == "__main__":
   
    # output_dir = 'test_data'
    # num = 1000 

    # output_dir = 'train_data'
    # num = 50000 
    # env = RectEnv()
    # env.main(output_dir, num)
    env = RectEnv()

    num = 3000 
    for i in range(10):
        output_dir = 'train_data_%02d' % i 
        env.main(output_dir, num)


