from PyQt5 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import sys
import pickle
import pandas as pd


from data_extraction import DataExtractor
from roboshifter import Roboshifter


class Window(QtWidgets.QWidget):
    FLAG_COLORS = {0: 'b', 1: 'r', 2: 'g', 3: 'y'}

    def __init__(self, data, labels, sf, result):
        QtWidgets.QWidget.__init__(self)

        self.data = data
        self.sf = sf
        self.index = int(0.55 * len(data))
        self.answers = {}
        self.result = result

        self.main_layout = QtWidgets.QVBoxLayout(self)

        self.text_box = QtWidgets.QLineEdit()
        f = self.text_box.font()
        f.setPointSize(18)
        self.text_box.setFont(f)

        self.figure = Figure(figsize=(20, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.button_group = QtWidgets.QButtonGroup()
        self.button_group.setExclusive(False)
        self.buttons = {}
        self.buttons_layout = QtWidgets.QHBoxLayout()
        for label in labels:
            self.buttons[label] = QtWidgets.QCheckBox(label)
            self.button_group.addButton(self.buttons[label])
            self.buttons_layout.addWidget(self.buttons[label])

        self.next_button = QtWidgets.QPushButton('Next')
        self.next_button.clicked.connect(self.next_obj)
        self.prev_button = QtWidgets.QPushButton('Prev')
        self.prev_button.clicked.connect(self.prev_obj)
        self.rule_buttons_layout = QtWidgets.QHBoxLayout()
        self.rule_buttons_layout.addWidget(self.prev_button)
        self.rule_buttons_layout.addWidget(self.next_button)

        self.progress_bar = QtWidgets.QProgressBar()

        self.main_layout.addWidget(self.text_box)
        self.main_layout.addWidget(self.canvas)
        self.main_layout.addLayout(self.buttons_layout)
        self.main_layout.addLayout(self.rule_buttons_layout)
        self.main_layout.addWidget(self.progress_bar)

        self.next_obj()

        self.show()

    def prev_obj(self):
        self.index -= 1

        self.draw_current()

    def next_obj(self):
        if self.index != -1:
            labels = [label for label, button in self.buttons.items()
                      if button.checkState() == QtCore.Qt.Checked]

            self.result[self.index] = (self.data[self.index][0], labels)

        # write_result(self.result)

        self.index += 1

        if self.index == len(self.data):
            self.close()
            return

        self.draw_current()

    def draw_current(self):
        key, data, default_labels = self.data[self.index]

        self.text_box.setText(str(key))

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        x, y = data.columns

        ax.scatter(data[x], data[y], c=sf.map(Window.FLAG_COLORS), alpha=0.2)

        self.canvas.draw()

        for button in self.buttons.values():
            button.setCheckState(QtCore.Qt.Unchecked)

        for default_label in default_labels:
            self.buttons[default_label].setCheckState(QtCore.Qt.Checked)

        self.progress_bar.setValue(float(self.index) / len(self.data) * 100)


def write_result(result):
    current = de.get_assesed_interactions()
    new = [a for a in result if a is not None]

    for key, value in new:
        current[key] = value

    with open(de.paths['ASSESED_INTERACTIONS'], 'wb') as outfile:
        pickle.dump(current, outfile)


def prepare_data(de):
    interactions = de.get_ht_interactions()

    rs = Roboshifter(verbose=False)
    labels = rs.INTERACTION_PREPARE.keys() + ['nothing']

    current = de.get_assesed_interactions()

    data = []

    df = de.get_train_data()

    histo_types = de.get_histo_types()
    for hk in de.get_histo_keys():
        ht = histo_types[hk]
        for key, value in interactions[ht].items():
            if len(key) == 2 and (key[0][0] != 'distance' and key[1][0] != 'distance' and key[1][1] != 'variational'):
                cols = [('my', ht, hk) + k for k in key]
                if all(col in df for col in cols):
                    new_key = (hk, key)
                    if new_key not in current or True:
                        data.append([new_key, df[cols], value])

    X, y = df.drop('flag', axis=1), df.flag
    rs.init_fit(X, y)
    rs.fit_stat_filter()
    stat_flag = rs.info[True]['stat_flag']
    sf = y.copy()
    sf[stat_flag == 1] += 2

    # sf = pd.Series(index=df.index)
    # index = df[df[('linear', 'switch')] == 7].index
    # sf.loc[:] = 0
    # sf.loc[index] = 1
    # sf[y == 1] += 2

    return data, labels, sf


if __name__ == '__main__':
    de = DataExtractor()

    data, labels, sf = prepare_data(de)
    result = [None] * len(data)

    app = QtWidgets.QApplication(sys.argv)

    win = Window(data, labels, sf, result)

    app.exec_()

    # write_result(result)