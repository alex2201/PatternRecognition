import numpy as np


class ClassifierClass:
    members = None

    def __init__(self, x_pos=0.0, y_pos=0.0, x_dispersion=1.0, y_dispersion=1.0):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_dispersion = x_dispersion
        self.y_dispersion = y_dispersion

    def generate_members(self, members_no):
        px, py = self.coords
        x_disp, y_disp = self.dispersion

        x_pts = np.random.uniform(px - x_disp, px + x_disp, members_no)
        y_pts = np.random.uniform(py - y_disp, py + y_disp, members_no)

        self.members = np.array(list(zip(x_pts, y_pts)))

    @property
    def coords(self):
        return self.x_pos, self.y_pos

    @property
    def dispersion(self):
        return self.x_dispersion, self.y_dispersion

    def __str__(self):
        return 'ClassifierClass'
