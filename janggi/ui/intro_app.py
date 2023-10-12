import sys
from PyQt5.QtWidgets import QPushButton, QLineEdit, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt

from ui.replay_viewer import ReplayViewer
from ui.game_player import GamePlayer

DEFAULT_GIB_FILE = "E:\work\Mandarin-AI\data\햇터몰배 제2회 세계인 장기대회 2(하여명 vs 전동하).gib"

class IntroApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        hboxes = []

        # hbox1
        hbox1 = QHBoxLayout()
        loadgibo_text = QLabel(self)
        loadgibo_text.setText("Gibo Replay")
        hbox1.addWidget(loadgibo_text)
        hboxes.append(hbox1)

        # hbox2
        hbox2 = QHBoxLayout()
        gibo_load_btn = QPushButton("Load",self)
        gibo_load_btn.clicked.connect(self.btn_gibo_load)    
        gibo_replay_start_btn = QPushButton("Replay Start",self)
        gibo_replay_start_btn.clicked.connect(self.btn_replay_start)    

        self.selected_file_text = QLineEdit(self)
        self.selected_file_text.setReadOnly(True)
        self.selected_file_text.setStyleSheet("color: #808080;" "background-color: #F0F0F0")
        hbox2.addWidget(self.selected_file_text)
        hbox2.addWidget(gibo_load_btn)
        hbox2.addWidget(gibo_replay_start_btn)
        hboxes.append(hbox2)

        # hbox3
        hbox3 = QHBoxLayout()
        new_game_start_btn = QPushButton("Start New Game",self)
        new_game_start_btn.clicked.connect(self.btn_newgame_start)    
        hboxes.append(hbox3)


        # vertical
        vbox = QVBoxLayout()
        vbox.addStretch(1)
        for hbox in hboxes:
            vbox.addLayout(hbox)
        vbox.addStretch(1)
        
        self.setLayout(vbox)

        self.setWindowTitle('Mandarin-AI')
        self.move(300, 300)
        self.resize(400, 200)
        self.setGeometry(800, 200, 500, 300)

        
        self.show()
        
   
    def btn_gibo_load(self):
        '''
            onClick - load gibo button
        '''
        fname=QFileDialog.getOpenFileName(self)    
        self.selected_file_text.setText(fname[0])

    def btn_replay_start(self):
        '''
            onClick - gibo replay start
        '''
        fname = self.selected_file_text.text()
        if not fname:
            fname = DEFAULT_GIB_FILE
        
        replay_viewer = ReplayViewer(fname)
        replay_viewer.run()

    def btn_newgame_start(self):
        game_player = GamePlayer()
        game_player.run()
