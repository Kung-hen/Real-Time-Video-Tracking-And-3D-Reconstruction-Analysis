from PyQt5 import QtWidgets
from controller import Form_controller

# cv2.imshow("aa", image_paths[0])
# cv2.show()
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = Form_controller()
    Form.show()
    sys.exit(app.exec_())
