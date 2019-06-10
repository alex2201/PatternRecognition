import numpy as np


class MemberGenerator:
    @staticmethod
    def generate_members(classes, members_no):
        for cls in classes:
            px, py = cls.coords
            x_disp, y_disp = cls.dispersion

            x_pts = np.random.uniform(px - x_disp, px + x_disp, members_no)
            y_pts = np.random.uniform(py - y_disp, py + y_disp, members_no)

            cls.members = np.array(list(zip(x_pts, y_pts)))
