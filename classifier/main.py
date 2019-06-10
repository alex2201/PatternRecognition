from tkinter import *
from tkinter import ttk

from Classifiers.BayesianClassifier import BayesianClassifier
from Classifiers.KNNClassifier import KNNClassifier
from Classifiers.ClassifierClass import ClassifierClass
from Classifiers.EuclideanClassifier import EuclideanClassifier
from Classifiers.MahalanobisClassifier import MahalanobisClassifier
from Generators.MemberGenerator import MemberGenerator


def generate_members():
    global obj_classes

    members_no = int(class_members_no_input.get())
    MemberGenerator.generate_members(classes, members_no)


def calculate_distances():
    global obj_classes
    global classifier_sel

    vx, vy = float(vector_x.get()), float(vector_y.get())

    classifier = classifiers_map[classifier_sel](classes)

    if classifier_sel != 'KNN':
        classifier.evaluate_vector((vx, vy))
    else:
        n = int(neighbors_no.get())
        classifier.evaluate_vector((vx, vy), n)


def save_class_config():
    sel_class.x_pos = float(config_x_coord.get())
    sel_class.x_dispersion = float(config_x_disp.get())
    sel_class.y_pos = float(config_y_coord.get())
    sel_class.y_dispersion = float(config_y_disp.get())


def clear_config_input():
    config_x_coord.delete(0, 'end')
    config_x_disp.delete(0, 'end')
    config_y_coord.delete(0, 'end')
    config_y_disp.delete(0, 'end')


def load_class_config():
    config_x_coord.insert(0, str(sel_class.x_pos))
    config_x_disp.insert(0, str(sel_class.x_dispersion))

    config_y_coord.insert(0, str(sel_class.y_pos))
    config_y_disp.insert(0, str(sel_class.y_dispersion))


def change_class_selection(event):
    global sel_class

    sel_class = obj_classes[int(class_sel.get()) - 1]

    clear_config_input()
    load_class_config()


def change_classifier_selection(event):
    global classifier_sel
    classifier_sel = classifier_combobox.get()


def generate_default_classes():
    global classes_indexes
    global obj_classes
    global sel_class

    sel_class = None
    class_no = int(class_no_input.get())
    classes_indexes, classes = [i for i in range(1, class_no + 1)], [ClassifierClass() for i in range(class_no)]
    class_sel.delete(0, 'end')

    clear_config_input()


def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    title_bar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + title_bar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(win_width, win_height, x, y))
    win.deiconify()


# PROPERTIES
obj_classes = []
sel_class = None
classes_indexes = []
classifiers = [
    'Euclidean',
    'Mahalanobis',
    'Bayesian',
    'KNN',
]
classifier_sel = classifiers[0]
classifiers_map = {
    'Euclidean': EuclideanClassifier,
    'Mahalanobis': MahalanobisClassifier,
    'Bayesian': BayesianClassifier,
    'KNN': KNNClassifier,
}

# GUI
window = Tk('MainScreen')
window.title('Classifier')
# window.geometry('800x600')
# center(window)

main_lbl = ttk.Label(text='Welcome to Vector Classifier!', font=(None, 18))
main_lbl.grid(row=0, column=1)

classifier_lbl = ttk.Label(text='Classifier')
classifier_lbl.grid(row=1, column=0)

classifier_combobox = ttk.Combobox(values=classifiers, justify='center')
classifier_combobox.current(0)
classifier_combobox.grid(row=1, column=1)
classifier_combobox.bind('<<ComboboxSelected>>', change_classifier_selection)

class_members_lbl = ttk.Label(text='Members')
class_members_lbl.grid(row=10, column=1)

class_members_no_lbl = ttk.Label(text='Number Members: ')
class_members_no_lbl.grid(row=11, column=0)

class_members_no_input = ttk.Spinbox(from_=0, to=100000, increment=1, wrap=True, justify='center')
class_members_no_input.set(0)
class_members_no_input.grid(row=11, column=1)

class_members_btn = ttk.Button(text='Generate', command=generate_members)
class_members_btn.grid(row=11, column=2)

class_no_lbl = ttk.Label(text='Number Classes: ')
class_no_lbl.grid(row=3, column=0)

class_no_input = ttk.Spinbox(from_=0, to=100, increment=1, wrap=True, justify='center')
class_no_input.set(0)
class_no_input.grid(row=3, column=1)

class_generate_btn = ttk.Button(text='Generate Classes', command=generate_default_classes)
class_generate_btn.grid(row=3, column=2)

class_sel_lbl = ttk.Label(text='Selected Class: ')
class_sel_lbl.grid(row=5, column=0)

class_sel = ttk.Combobox(values=classes_indexes, justify='center',
                         postcommand=lambda: class_sel.configure(values=classes_indexes))
class_sel.bind('<<ComboboxSelected>>', change_class_selection)
class_sel.grid(row=5, column=1)

config_lbl = ttk.Label(text='Class Configuration')
config_lbl.grid(row=4, column=1)

config_x_coord_lbl = ttk.Label(text='x coords')
config_x_coord_lbl.grid(row=6, column=0)

config_btn = ttk.Button(text='Save Class Config', command=save_class_config)
config_btn.grid(row=6, column=2, rowspan=4)

config_x_coord = ttk.Entry()
config_x_coord.grid(row=6, column=1)

config_x_disp_lbl = ttk.Label(text='x dispersion')
config_x_disp_lbl.grid(row=7, column=0)

config_x_disp = ttk.Entry()
config_x_disp.grid(row=7, column=1)

config_y_coord_lbl = ttk.Label(text='y coords')
config_y_coord_lbl.grid(row=8, column=0)

config_y_coord = ttk.Entry()
config_y_coord.grid(row=8, column=1)

config_y_disp_lbl = ttk.Label(text='y dispersion')
config_y_disp_lbl.grid(row=9, column=0)

config_y_disp = ttk.Entry()
config_y_disp.grid(row=9, column=1)

sep = ttk.Label(text='Vector Coordinates', justify='center')
sep.grid(row=12, column=1)
sep = ttk.Label(text='#Neighbors', justify='center')
sep.grid(row=12, column=2)

vector_x_lbl = ttk.Label(text='vector x')
vector_x_lbl.grid(row=13, column=0)

vector_x = ttk.Entry()
vector_x.insert(0, '0')
vector_x.grid(row=13, column=1)

neighbors_no = ttk.Entry()
neighbors_no.insert(0, '0')
neighbors_no.grid(row=13, column=2)

vector_y_lbl = ttk.Label(text='vector y')
vector_y_lbl.grid(row=14, column=0)

vector_y = ttk.Entry()
vector_y.insert(0, '0')
vector_y.grid(row=14, column=1)

calculate_btn = ttk.Button(text='Calculate', command=calculate_distances)
calculate_btn.grid(row=15, column=1)

mainloop()
