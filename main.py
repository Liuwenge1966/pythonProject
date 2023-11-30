"""
This is the main module.
"""
# coding=utf-8
from PyQt5 import QtCore, QtGui, QtWidgets
from MainWidget import CamShow
import sys

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_widget = CamShow()  # QtWidgets.QMainWindow()
    main_widget.setWindowTitle("version: " + "1.0.0.1")
    main_widget.show()
    # deskRect = QtWidgets.QApplication.desktop().screenGeometry()
    # x = deskRect.width() / 12
    # y = deskRect.height() / 12
    # main_widget.move(x, y)
    # main_widget.resize(deskRect.right() - 2 * x, deskRect.bottom() - 2 * y)
    sys.exit(app.exec_())
