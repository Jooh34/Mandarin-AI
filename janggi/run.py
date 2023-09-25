import sys
from PyQt5.QtWidgets import QApplication, QWidget

from ui.intro_app import IntroApp

def run_intro_app():
    app = QApplication(sys.argv)
    window = IntroApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_intro_app()